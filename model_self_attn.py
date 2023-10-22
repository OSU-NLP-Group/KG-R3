import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from functools import partial
import numpy as np

from bert.modeling_bert import BertEncoder, BertPreTrainedModel
from transformers.models.bert.configuration_bert import BertConfig


class GraphTransformer(nn.Module):
	config_class = BertConfig

	def __init__(self, n_entities, n_relations, args):
		super(GraphTransformer, self).__init__()
		self.n_entities = n_entities
		self.n_relations = n_relations
		self.embed_dim = args.embed_dim
		self.args = args
		self.label_smoothing = args.label_smoothing

		# initialize entity lookup embedding
		self.ent_embedding = nn.Embedding(self.n_entities, self.embed_dim)

		# initialize relation lookup embedding
		if not self.args.add_inverse_rels:
			self.rel_embedding = nn.Embedding(self.n_relations, self.embed_dim)
		else:
			self.rel_embedding = nn.Embedding(self.n_relations*2, self.embed_dim)

		# initialize token type embedding for subgraph
		self.type_embeds = nn.Embedding(2, self.embed_dim)

		# initialize segment embedding
		if self.args.add_segment_embed:
			self.segment_embeddings = nn.Embedding(2, self.embed_dim)

		config = BertConfig(0, hidden_size=self.embed_dim,
							num_hidden_layers=args.n_bert_layers,
							num_attention_heads=args.n_attn_heads,
							intermediate_size=1280,
							hidden_act='gelu',
							hidden_dropout_prob=args.hidden_dropout_prob,
							attention_probs_dropout_prob=args.attention_probs_dropout_prob,
							max_position_embeddings=0,  # no effect
							type_vocab_size=0,  # no effect
							initializer_range=0.02)

		# initialize subgraph self-attention encoder
		self.subgraph_self_attn_encoder = BertEncoder(config)
		self.subgraph_self_attn_encoder.config = config
		self.subgraph_self_attn_encoder_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-12)
		
		# initialize query self-attention encoder parameters
		self.cls_query = nn.Parameter(torch.Tensor(1, self.embed_dim))
		self.type_embeds_query = nn.Embedding(3, self.embed_dim)
		self.layer_norm_query = nn.LayerNorm(self.embed_dim, eps=1e-12)
		self.self_attention_encoder_query = BertEncoder(config)
		self.self_attention_encoder_query.config = config

		# projection layer for concatenated embed
		self.W = nn.Linear(2*self.embed_dim, self.embed_dim)
		
		self.loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='mean', label_smoothing=self.label_smoothing)

		self.init_weights()

	def init_weights(self):
		nn.init.normal_(self.ent_embedding.weight.data, 0.0, 0.02)
		nn.init.normal_(self.rel_embedding.weight.data, 0.0, 0.02)
		nn.init.normal_(self.type_embeds.weight, std=0.02)

		nn.init.xavier_uniform_(self.W.weight)
		torch.nn.init.normal_(self.cls_query, std=0.02)
		nn.init.normal_(self.type_embeds_query.weight, std=0.02)

		if self.args.add_segment_embed:
			nn.init.normal_(self.segment_embeddings.weight, std=0.02)

		self.subgraph_self_attn_encoder.apply(partial(BertPreTrainedModel._init_weights, self.subgraph_self_attn_encoder))
		self.self_attention_encoder_query.apply(partial(BertPreTrainedModel._init_weights, self.self_attention_encoder_query))

	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		token_type_ids=None,
		position_ids=None,
		inputs_embeds=None,
		ent_masked_lm_labels=None, 
		n_entity_nodes=None,
		ent_ids=None, # source entity
		rel_ids=None, # query relation
		ent_len_tensor=None,
	):

		n_entity_nodes = int(ent_len_tensor.max())
		batch_size = input_ids.size(0)

		# compute subgraph lookup embeds
		ent_embeddings = self.ent_embedding(
				input_ids[:, :n_entity_nodes])

		rel_embeddings = self.rel_embedding(
			input_ids[:, n_entity_nodes:])

		inputs_embeds = torch.cat([ent_embeddings, rel_embeddings],
								  dim=1)  # [batch, seq_len, hidden_size]
		
		# compute component embeddings for subgraph embedding layer
		pos = self.type_embeds(token_type_ids)
		inputs_embeds = inputs_embeds + pos

		if self.args.add_segment_embed:
			segment_embeddings = self.segment_embeddings(segment_ids)
			inputs_embeds += segment_embeddings

		inputs_embeds = F.dropout(inputs_embeds, p=0.6, training=self.training)

		inputs_embeds = self.subgraph_self_attn_encoder_layer_norm(inputs_embeds) # culprit line

		# transform attention mask for input to subgraph self-attention encoder
		extended_attention_mask = attention_mask[:, None, :, :]
		extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

		# compute subgraph self-attention encoder representations
		subgraph_embed = self.subgraph_self_attn_encoder(inputs_embeds, attention_mask=extended_attention_mask, n_entity_nodes=n_entity_nodes, return_dict=False)[0][:, 0]
		
		# compute query lookup embeds
		src_ent_embed = self.ent_embedding(ent_ids) # [batch, hid_dim]
		query_rel_embed = self.rel_embedding(rel_ids) # [batch, hid_dim]

		# compute component embeddings for query embedding layer
		query_embed = torch.stack([src_ent_embed, query_rel_embed], dim=1) # [batch, 2, hid_dim]
		query_embed = torch.cat([self.cls_query.expand(query_embed.size(0), 1, query_embed.size(-1)), query_embed], dim=1) # [batch, 3, hid_dim]
		pos = self.type_embeds_query(torch.arange(0, 3, device=ent_ids.get_device())).unsqueeze(0).repeat(query_embed.shape[0], 1, 1)
		query_embed = query_embed + pos
		query_embed = F.dropout(query_embed, p=0.6, training=self.training)
		query_embed = self.layer_norm_query(query_embed) # [batch, 2, hid_dim]

		# transform attention mask for input to query self-attention encoder
		attention_mask = torch.ones(query_embed.size(0), query_embed.size(1), query_embed.size(1), device=ent_ids.get_device()) # [batch, 3, 3]
		extended_attention_mask = (1.0 - attention_mask.unsqueeze(1)) * -10000.0

		# compute query self-attention encoder representations
		query_embed = self.self_attention_encoder_query(query_embed, attention_mask=extended_attention_mask, return_dict=False)[0][:, 0] # [batch, hid_dim]		

		# compute final logits
		final_embed = self.W(torch.cat([query_embed, subgraph_embed], dim=-1)) # [batch, hid_dim]

		o_emb = self.ent_embedding.weight
		o_emb = F.dropout(o_emb, p=0.6, training=self.training)
		ent_logits = torch.mm(final_embed, o_emb.transpose(1, 0))

		ent_predict = torch.argmax(ent_logits, dim=-1)

		ent_masked_lm_loss = self.loss_fct(ent_logits.view(-1, ent_logits.size(-1)), ent_masked_lm_labels.view(-1))
		loss = ent_masked_lm_loss

		return_dict = {'loss': loss,
				'ent_logits': ent_logits,
				'entity_pred': ent_predict,
				}
				
		return return_dict

