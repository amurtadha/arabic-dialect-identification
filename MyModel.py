import torch
import torch.nn as nn
from transformers import  AutoModel, AutoConfig, AutoModelForSequenceClassification
import torch.nn.functional as F


class Attn(nn.Module):
    def __init__(self, h_dim):
        super(Attn, self).__init__()
        self.h_dim = h_dim
        self.cr_att = nn.Sequential(
            nn.Linear(h_dim, 24),
            nn.ReLU(True),
            nn.Linear(24, 1)
        )

    def forward(self, encoder_outputs):
        b_size = encoder_outputs.size(0)
        s_size = encoder_outputs.size(1)
        # attn_ene = self.main(encoder_outputs.view(-1, self.h_dim))  # (b, s, h) -> (b * s, 1)
        # return F.softmax(attn_ene.view(b_size, -1), dim=1).unsqueeze(2)  # (b*s, 1) -> (b, s, 1)

        attn_ene = self.cr_att(encoder_outputs.reshape(b_size*s_size,self.h_dim))  # (b, s, h) -> (b * s, 1)
        return F.softmax(attn_ene.reshape(b_size,s_size), dim=1).unsqueeze(2)  # (b*s, 1) -> (b, s, 1)

class EncoderRNN(nn.Module):
    def __init__(self,embedding_matrix, emb_dim, h_dim, gpu=True, v_vec=None, batch_first=True):
        super(EncoderRNN, self).__init__()
        self.gpu = gpu
        self.h_dim = h_dim
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        # self.embed = nn.Embedding(v_size, emb_dim)
        # if v_vec is not None:
        #     self.embed.weight.data.copy_(v_vec)
        self.lstm = nn.LSTM(emb_dim, h_dim, batch_first=batch_first,
                            bidirectional=False)

    # def init_hidden(self, b_size):
    #     h0 = Variable(torch.zeros(1 * 2, b_size, self.h_dim))
    #     c0 = Variable(torch.zeros(1 * 2, b_size, self.h_dim))
    #     if self.gpu:
    #         h0 = h0.cuda()
    #         c0 = c0.cuda()
    #     return (h0, c0)

    def forward(self, sentence, lengths=None):
        # self.hidden = self.init_hidden(sentence.size(0))
        # emb = self.embed(sentence)
        # packed_emb = emb
        #
        # if lengths is not None:
        #     lengths = lengths.view(-1).tolist()
        #     packed_emb = nn.utils.rnn.pack_padded_sequence(emb, lengths)
        #
        # out, hidden = self.lstm(packed_emb, self.hidden)

        #
        # if lengths is not None:
        #     out = nn.utils.rnn.pad_packed_sequence(output)[0]
        #
        # out = out[:, :, :self.h_dim] + out[:, :, self.h_dim:]
        emb = self.embed(sentence)
        out, hidden = self.lstm(emb, self.hidden)
        return out


class AttnClassifier(nn.Module):
    def __init__(self, opt,embedding_matrix, batch_first=True):
        super(AttnClassifier, self).__init__()
        self.attn = Attn(opt.hidden_dim*2)


        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))

        self.lstm = nn.LSTM(opt.hidden_dim, opt.hidden_dim, batch_first=batch_first,
                            bidirectional=True)

        self.cr = nn.Linear(opt.hidden_dim*2, opt.lebel_dim)
        # self.encoder= EncoderRNN(embedding_matrix, opt.hidden_dim, opt.hidden_dim)

    def forward(self, inputs):
        emb = self.embed(inputs[0])
        out, hidden = self.lstm(emb)
        # encoder_outputs= self.encoder(inputs[0])
        attns = self.attn(out)  # (b, s, 1)
        feats = (out * attns).sum(dim=1)  # (b, s, h) -> (b, h)
        # return F.log_softmax(self.main(feats), dim=-1)
        return self.cr(feats)
class PureLSTMClassifier(nn.Module):
    def __init__(self, opt,embedding_matrix):
        super(PureLSTMClassifier, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        # self.attn = Attn(opt.hidden_dim)
        self.cr = nn.Linear(opt.hidden_dim, opt.lebel_dim)
        self.encoder= nn.LSTM( opt.hidden_dim, opt.hidden_dim, batch_first=True)

    def forward(self, inputs):
        embed = self.embed(inputs[0])
        output, (final_hidden_state, final_cell_state)= self.encoder(embed)
        # return F.log_softmax(self.cr(final_hidden_state[-1]), dim=-1)
        return self.cr(final_hidden_state[-1])

class PureBERT(nn.Module):

    def __init__(self, args):
        super(PureBERT, self).__init__()
        config = AutoConfig.from_pretrained(args.pretrained_bert_name)
        config.num_labels = args.lebel_dim
        self.encoder = AutoModelForSequenceClassification.from_pretrained(args.pretrained_bert_name, config=config)
        self.encoder.to('cuda')
    def forward(self, inputs):
        input_ids, token_type_ids, attention_mask = inputs[:3]
        outputs = self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        return outputs['logits']

class ADIBERTCustom(nn.Module):

    def __init__(self, args, hidden_size=256):
        super(ADIBERTCustom, self).__init__()
        config = AutoConfig.from_pretrained(args.pretrained_bert_name)
        self.enocder = AutoModel.from_pretrained(args.pretrained_bert_name, config=config)
        self.enocder.to('cuda')

        # self.lstm = nn.LSTM(args.hidden_dim, args.hidden_dim)
        self.attn = Attn(args.hidden_dim)

        layers = [nn.Linear(config.hidden_size, args.lebel_dim)]
        self.classifier = nn.Sequential(*layers)

    def forward(self, inputs):

        input_ids,token_type_ids, attention_mask = inputs[:3]
        # with torch.no_grad():
        outputs = self.enocder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            # pooled_output = outputs['last_hidden_state'][:, 0, :]
        pooled_output = outputs['last_hidden_state']

        # output, (final_hidden_state, final_cell_state) = self.lstm(pooled_output)
        attns = self.attn(pooled_output)  # (b, s, 1)
        feats = (pooled_output * attns).sum(dim=1)  # (b, s, h) -> (b, h)
        logits = self.classifier(feats)
        # logits = self.classifier(pooled_output)

        return logits