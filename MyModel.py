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
        attn_ene = self.cr_att(encoder_outputs.reshape(b_size*s_size,self.h_dim))  # (b, s, h) -> (b * s, 1)
        return F.softmax(attn_ene.reshape(b_size,s_size), dim=1).unsqueeze(2)  # (b*s, 1) -> (b, s, 1)

class AttnClassifier(nn.Module):
    def __init__(self, opt,embedding_matrix, batch_first=True):
        super(AttnClassifier, self).__init__()
        self.attn = Attn(opt.hidden_dim*2)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = nn.LSTM(opt.hidden_dim, opt.hidden_dim, batch_first=batch_first,bidirectional=True)
        self.cr = nn.Linear(opt.hidden_dim*2, opt.lebel_dim)       
    def forward(self, inputs):
        emb = self.embed(inputs[0])
        out, hidden = self.lstm(emb)
        attns = self.attn(out)  # (b, s, 1)
        feats = (out * attns).sum(dim=1)  # (b, s, h) -> (b, h)
        return self.cr(feats)
    

class ADIBERTCustom(nn.Module):
    def __init__(self, args, hidden_size=256):
        super(ADIBERTCustom, self).__init__()
        config = AutoConfig.from_pretrained(args.pretrained_bert_name)
        self.enocder = AutoModel.from_pretrained(args.pretrained_bert_name, config=config)
        self.enocder.to('cuda')
        self.attn = Attn(args.hidden_dim)
        layers = [nn.Linear(config.hidden_size, args.lebel_dim)]
        self.classifier = nn.Sequential(*layers)

    def forward(self, inputs):
        input_ids,token_type_ids, attention_mask = inputs[:3]
        outputs = self.enocder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = outputs['last_hidden_state']
        attns = self.attn(pooled_output)  # (b, s, 1)
        feats = (pooled_output * attns).sum(dim=1)  # (b, s, h) -> (b, h)
        logits = self.classifier(feats)
        return logits
