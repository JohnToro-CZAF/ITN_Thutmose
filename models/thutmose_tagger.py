from time import perf_counter
from typing import Dict, List, Optional

import torch
from torch.nn import CrossEntropyLoss
from omegaconf import DictConfig
from pytorch_lightning import Trainer

class ThutmoseTaggerModel(NLPModel):
    """
    BERT-based tagging model for ITN, inspired by LaserTagger approach.
    It maps spoken-domain input words to tags:
        KEEP, DELETE, or any of predefined replacement tags which correspond to a written-domain fragment.
    Example: one hundred thirty four -> _1 <DELETE> 3 4_
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None) -> None:
        super().__init__(cfg=cfg, trainer=trainer)
        label_map_file = cfg.label_map
        semiotic_classes_file = cfg.semiotic_classes
        self.label_map = read_label_map(label_map_file)
        self.semiotic_classes = read_semiotic_classes(semiotic_classes_file)

        self.num_labels = len(self.label_map)
        self.num_semiotic_labels = len(self.semiotic_classes)

        self.id_2_tag = {tag_id: tagging.Tag(tag) for tag, tag_id in self.label_map.items()}
        self.id_2_semiotic = {semiotic_id: semiotic for semiotic, semiotic_id in self.semiotic_classes.items()}

        self.max_sequence_len = cfg.get("max_sequence_len", self.tokenizer.tokenizer.model_max_length)

        # setup to trach metrics
        # we will have (len(self.semiotic_classes) + 1) labels
        # last one stands for WRONG (span in which the predicted tags dont match the labels)
        # this is need to feed the sequence of classes to classification_report during the validation

        label_ids = self.semiotic_classes.copy()
        label_ids["WRONG"] = len(self.semiotic_classes)
        
        self.hidden_size = cfg.hidden_size
        self.logits = TokenClassifier(
            self.hidden_size, num_classes=self.num_labels, num_layers=, log_softmax=False, dropout=0.1
        )
        self.semiotic_logits = TokenClassifier(
            self.hidden_size, num_classes=self.num_semiotic_labels, num_layers=1, log_softmax=False, dropout=0.1
        )

        self.loss_fn = CrossEntropyLoss(weight=None, reduction='mean', ignore_index=-100)
        self.builder = bert_example.BertExampleBuilder(
            self.label_map, self.semiotic_classes, self.tokenizr.tokenizer, self.max_sequence_len
        )

    def foward(self, input_ids, input_mask, segment_ids):
        src_hiddens = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        logits = self.logits(hidden_states=src_hiddens)
        semiotic_logits = self.semiotic_logits(hidden_states=src_hiddens)
        return tag_logits, semiotic_logits
    
    #Training
    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        """
        input_ids, input_mask, segment_ids, label_mask, labels, semiotic_labels, _ = batch
        tag_logits, semiotic_logits = self.foward(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
        loss_on_tags = self.loss_fn(logits=tag_logits, labels=labels, loss_mask=labels_mask)
        loss_on_semiotic = self.loss_fn(logits=semiotic_logits, labels=semiotic_labels, loss_mask=label_mask)
        loss = loss_on_tags + loss_on_semiotic
        lr = self._optimizer.param_group[0]['lr']
        print("train_loss", loss)
        return {'loss': loss, 'lr': lr}
    
    def validation_step(self, batch, batch_idx):
        