import copy
import torch
import numpy as np
from torch import nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertPooler
from torchcrf import CRF
from torch.nn import CrossEntropyLoss, MSELoss
from utils.layers import Highway, SelfAttention, PicoAttention, Boom
from utils.layers import MultiHeadedAttention, PositionwiseFeedForward, Encoder,EncoderLayer, Decoder,DecoderLayer

class BertForSequenceTagging(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        pico_embeddings_size = 100  # same as the vocab size - we don't care about this at this point
        self.bert = BertModel(config)

        self.pico_embeddings = nn.Embedding(pico_embeddings_size,pico_embeddings_size) #randomly initialized

        self.crf = CRF(4, batch_first=True) #since we removed the 2 labels
        self.classifier = nn.Linear(2 * config.hidden_size, config.num_labels)

        N = 4 #layer number
        h = 4 #heads
        dropout_value = 0.1
        d_model = config.hidden_size+pico_embeddings_size
        d_ff = 2048
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, dropout=dropout_value)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout=dropout_value)
        self.encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout_value), N)
        self.decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout=dropout_value ), N)


        self.classifier_bienc = nn.Linear(2*d_model,config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        pico=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        pico_input = self.pico_embeddings(pico)
        latent_input = torch.cat([sequence_output, pico_input], dim=-1)
        forward_encode = self.encoder(latent_input, None)
        # backward_latent_input = torch.flip(latent_input, [-1])
        # backward_encode = self.encoder(backward_latent_input, None)
        encode = torch.cat([forward_encode,backward_encode], dim=-1)
        emissions = self.classifier_bienc(encode)

        if labels is not None:
            loss = self.crf(emissions, labels)

            path = self.crf.decode(emissions)
            path = torch.LongTensor(path)

            return (-1*loss, emissions, path)
        else:
            path = self.crf.decode(emissions)
            path = torch.LongTensor(path)

            return path

class BertForSeqClass(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        pico_embeddings_size = 100 
        pico_rnn_cell_size = 100 


        # encoder
        self.bert = BertModel(config)

        #pico embeddings from pretrained EBM model 
        self.pico_embeddings = nn.Embedding(5,pico_embeddings_size)

        # transformer encoder/decoder
        N = 4  # layer number
        h = 4  # heads
        dropout_value = 0.1
        d_model = config.hidden_size + pico_embeddings_size
        d_ff = 2*d_model
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, dropout=dropout_value)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout=dropout_value)
        self.encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout_value), N)
        self.decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout=dropout_value), N)

        # past-transformer pooling
        self.transformer_pooler = nn.Linear(config.hidden_size + pico_embeddings_size,config.hidden_size + pico_embeddings_size)
        self.transformer_pooler_activation = nn.Tanh()

        # relation classification
        self.classifier = nn.Linear(config.hidden_size + pico_embeddings_size, config.num_labels)  
        

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids = None,
                head_mask = None, inputs_embeds = None, pico=None, labels=None, task=None, return_dict=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        
        return_dict = return_dict if return_dict is not None else True

        sequence_embeddings = outputs[0]
        pico_embeddings = self.pico_embeddings(pico)
        input = torch.cat([sequence_embeddings,pico_embeddings], dim=-1)

        latent_input = self.encoder(latent_input, None)
        latent_input = self.decoder(latent_input, latent_input, None, None)

        latent_pooled = latent_input[:,0]
        latent_pooled = self.transformer_pooler(latent_pooled)
        latent_pooled = self.transformer_pooler_activation(latent_pooled)

        logits = self.classifier(latent_pooled)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return loss, logits 

class BertForPicoSequenceTagging(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)

        self.crf = CRF(config.num_labels, batch_first=True)

        self.classifier_bienc = nn.Linear(2*config.hidden_size,config.num_labels)

        N = 4 #layer number
        h = 4 #heads
        dropout_value = 0.1
        d_model = config.hidden_size
        d_ff = 2048
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, dropout=dropout_value)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout=dropout_value)
        self.encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout_value), N)
        self.decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout=dropout_value ), N)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        pico=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        forward = sequence_output
        backward = torch.flip(sequence_output,[-1])
        enc_forward = self.encoder(forward, None)
        enc_backward = self.encoder(backward, None)
        bienc = torch.cat([enc_forward,enc_backward], dim=-1)
        emissions = self.classifier_bienc(bienc)


        if labels is not None:
            loss = self.crf(emissions, labels)

            path = self.crf.decode(emissions)
            path = torch.LongTensor(path)

            return (-1*loss, emissions, path)
        else:
            path = self.crf.decode(emissions)
            path = torch.LongTensor(path)

            return path