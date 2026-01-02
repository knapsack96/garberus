import torch
import torch.nn as nn
from transformers import TimesformerForVideoClassification

class MultiHeadTimesformer(nn.Module):
    def __init__(self):
        super().__init__()
        # backbone pre-addestrata su Kinetics-600
        self.backbone = TimesformerForVideoClassification.from_pretrained(
            "facebook/timesformer-hr-finetuned-k600",
            trust_remote_code=True,
            use_safetensors=True
        )
        embed_dim = 768
        if hasattr(self.backbone, "classifier") and hasattr(self.backbone.classifier, "out_features"):
            embed_dim = self.backbone.classifier.out_features

        # teste separate
        self.dump_head = nn.Linear(embed_dim, 2)
        self.timestamp_head = nn.Linear(embed_dim, 2)

    def _extract_feat(self, outputs):
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state.mean(dim=1)
        return outputs.logits

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values, return_dict=True)
        feat = self._extract_feat(outputs)
        return self.dump_head(feat), self.timestamp_head(feat)

# ================= Checkpoint delle teste =================
checkpoint_dump = "checkpoint_extradati_solo_dump1.pth"
checkpoint_ts   = "checkpoint_9.pth"

state_dump = torch.load(checkpoint_dump, map_location="cpu")
state_ts   = torch.load(checkpoint_ts, map_location="cpu")

# ================= Crea modello =================
model = MultiHeadTimesformer()

# Carica le teste dai checkpoint
model.dump_head.load_state_dict({
    k.replace("dump_head.", ""): v
    for k,v in state_dump.items() if k.startswith("dump_head.")
})

model.timestamp_head.load_state_dict({
    k.replace("timestamp_head.", ""): v
    for k,v in state_ts.items() if k.startswith("timestamp_head.")
})

# ================= Salva tutto in un unico checkpoint =================
torch.save(model.state_dict(), "garberus_full.pth")
print("✅ Modello completo salvato come garberus_full.pth")
