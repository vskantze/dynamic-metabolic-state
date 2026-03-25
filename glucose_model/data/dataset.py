import jax.numpy as jnp

class GlucoseDataset:
    def __init__(self, meals):
        self.meals = meals

    def __len__(self):
        return len(self.meals)

    def __getitem__(self, idx):
        m = self.meals[idx]

        return {
            "person_id": m["person_id"],
            "person_idx": m["person_idx"],
            "meal": {
                "carbs": jnp.array(m["meal"]["carbs"]),
                "fat": jnp.array(m["meal"]["fat"]),
                "protein": jnp.array(m["meal"]["protein"]),
            },
            "context": jnp.array(m["context"]),
            "time": jnp.array(m["time"]),
            "glucose": jnp.array(m["glucose"]),
            "baseline": m["glucose"][0]
        }
    

def create_batch(dataset, indices):
    batch = [dataset[i] for i in indices]

    return {
        "meal": {
            "carbs": jnp.stack([b["meal"]["carbs"] for b in batch]),
            "fat": jnp.stack([b["meal"]["fat"] for b in batch]),
            "protein": jnp.stack([b["meal"]["protein"] for b in batch]),
        },
        "context": jnp.stack([b["context"] for b in batch]),
        "time": jnp.stack([b["time"] for b in batch]),
        "glucose": jnp.stack([b["glucose"]-b["baseline"]  for b in batch]),
        "baseline": jnp.stack([b["baseline"] for b in batch]),
        "person_id": jnp.stack([b["person_id"] for b in batch]),
        "person_idx": jnp.stack([b["person_idx"] for b in batch])
    }