
import torch
from dataclasses import dataclass

import transformers
from transformers.utils import ModelOutput

@dataclass
class DecoderModelOutput(ModelOutput):
    logits: torch.FloatTensor = None
    loss: torch.FloatTensor = None
    hidden_states : torch.FloatTensor = None

class DecoderConfig(transformers.PretrainedConfig):
    model_type = "router"
    def __init__(
            self,
            vocab_size : int = 32000,
            embedding_dim : int = 2048,
            num_heads : int = 2,
            padding_idx : int = 0,
            output_hidden_states : bool = False 

    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.padding_idx = padding_idx
        self.output_hidden_states = output_hidden_states

class DecoderModel(transformers.PreTrainedModel):
    def __init__(
            self, 
            config : DecoderConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.output_hidden_states = config.output_hidden_states
        self.embedding = torch.nn.Embedding(config.vocab_size, config.embedding_dim, config.padding_idx)
        self.transformer_decoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(config.embedding_dim, nhead=config.num_heads),
            num_layers=1
        )
        self.fc = torch.nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask = None, labels = None) -> DecoderModelOutput:  
        hidden_states = None
        embedded = self.embedding(input_ids)
        seq_length = input_ids.shape[1]
        attn_mask = torch.tril(torch.ones((seq_length, seq_length)))
        output = self.transformer_decoder(embedded, mask = attn_mask, is_causal = True)
        if self.output_hidden_states:
            hidden_states = output
        output = self.fc(output)
        logits = output.float()

        loss = None
        shift_labels = input_ids[..., 1:].contiguous()

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        # Flatten the tokens

        loss_fct = torch.nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

        return DecoderModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states
        )
    
    def get_input_embeddings(self):
        return self.embedding

    



