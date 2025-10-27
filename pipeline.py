import os
from utils import MaterialDataProcessor, MaterialDataset, create_dataloader

if __name__ == "__main__":
    api_key = "u8sNESP1M49d1bK6fC0jFor7O8ht1oXw"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(current_dir, "Data"), exist_ok=True)
    h5_path = os.path.join(current_dir, "Data", "materials_data.h5")
    
    processor = MaterialDataProcessor(api_key, disable_progress=False)
    processor.pipeline(h5_path)
    
    dataset = MaterialDataset(h5_path)
    dataloader = create_dataloader(dataset, batch_size=2)
    batch, _ = next(iter(dataloader))

    print(f"Frequencies Shape: {batch['frequencies'].shape}")