import torch
import numpy as np
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from PIL import Image
import torchvision.transforms as transforms
from safetensors.torch import load_file
import argparse

class SDDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Directory not found: {data_dir}")
            
        self.image_paths = list(self.data_dir.glob("*.jpg")) + list(self.data_dir.glob("*.png"))
        if not self.image_paths:
            raise ValueError(f"No images (*.jpg, *.png) found in {data_dir}")
            
        print(f"Found {len(self.image_paths)} images")
        
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        self.transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Fixed: Use 3-channel normalization
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Remove noise channel addition - we'll handle latents in find_lr
        # Normalize image to [-1, 1] range
        image = 2 * image - 1
        
        # Use filename as caption (without extension)
        caption = img_path.stem.replace('_', ' ')
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'pixel_values': image,
            'input_ids': tokens.input_ids[0]
        }

class SDLRFinder:
    def __init__(self, model_path="SD", device="cuda"):
        self.device = device
        model_path = Path(model_path)

        # Пути к файлам
        unet_path = model_path / "unet" / "diffusion_pytorch_model.fp16.safetensors"
        text_encoder_path = model_path / "text_encoder" / "model.fp16.safetensors"
        text_encoder_config_path = model_path / "text_encoder" / "config.json"

        # Проверяем наличие файлов
        if not unet_path.exists():
            raise ValueError(f"UNet model not found at {unet_path}")
        if not text_encoder_path.exists():
            raise ValueError(f"Text encoder model not found at {text_encoder_path}")
        if not text_encoder_config_path.exists():
            raise ValueError(f"Text encoder config not found at {text_encoder_config_path}")

        # Создаем базовую модель UNet с правильной конфигурацией для SD 1.5
        self.unet = UNet2DConditionModel(
            sample_size=64,  # Размер латентного пространства
            in_channels=4,   # Количество входных каналов
            out_channels=4,  # Количество выходных каналов
            layers_per_block=2,  # Слои в каждом блоке
            block_out_channels=(320, 640, 1280, 1280),  # Каналы в блоках
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            cross_attention_dim=768,  # Размерность для cross attention (SD 1.5)
        ).to(device, torch.float16)
        
        # Загружаем конфиг и создаем text encoder с правильной конфигурацией
        from transformers import CLIPConfig
        text_encoder_config = CLIPConfig.from_json_file(text_encoder_config_path)
        text_encoder_config.text_config.hidden_size = 768  # SD 1.5 specific
        self.text_encoder = CLIPTextModel(text_encoder_config.text_config).to(device, torch.float16)
        
        # Загружаем веса из safetensors напрямую
        unet_state_dict = load_file(str(unet_path))
        text_encoder_state_dict = load_file(str(text_encoder_path))
        
        # Убираем prefix "text_model." из ключей для text encoder
        cleaned_text_encoder_state_dict = {}
        for k, v in text_encoder_state_dict.items():
            if k.startswith("text_model."):
                cleaned_text_encoder_state_dict[k.replace("text_model.", "")] = v
            else:
                cleaned_text_encoder_state_dict[k] = v
        
        # Загружаем веса в модели
        self.unet.load_state_dict(unet_state_dict)
        self.text_encoder.load_state_dict(cleaned_text_encoder_state_dict, strict=False)
        
        # Оптимизация памяти
        self.unet = self.unet.to(memory_format=torch.channels_last)
        self.text_encoder = self.text_encoder.to(memory_format=torch.channels_last)
        
        self.timesteps = torch.linspace(0, 999, 1000, dtype=torch.long).to(device)
        self.history = {"lr": [], "loss": []}
        self.best_loss = float('inf')

    def find_lr(self, train_loader, optimizer, criterion,
                min_lr=1e-04, max_lr=1, num_iter=100):
        # Сбрасываем генератор случайных чисел в начале поиска
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        lrs = []
        losses = []
        
        # Log-spaced learning rates
        lr_factor = (max_lr / min_lr) ** (1 / num_iter)
        lr = min_lr
        
        for i in tqdm(range(num_iter)):
            optimizer.param_groups[0]['lr'] = lr
            
            # Get batch
            batch = next(iter(train_loader))
            inputs, labels = batch['input_ids'].to(self.device), batch['pixel_values'].to(self.device)
            
            # Forward pass
            with torch.no_grad():
                text_embeddings = self.text_encoder(inputs)[0]
            
            optimizer.zero_grad()
            
            # Convert RGB image to latent space dimensions with correct dtype
            latent_height = labels.shape[2] // 8
            latent_width = labels.shape[3] // 8
            latents = torch.randn(
                labels.shape[0], 4, latent_height, latent_width, 
                device=self.device, dtype=torch.float16  # Explicitly set dtype
            )
            target_latents = torch.randn_like(latents)
            
            timestep = self.timesteps[0].expand(labels.shape[0])
            
            # Ensure all inputs are float16
            inputs = inputs.to(dtype=torch.float16)
            text_embeddings = text_embeddings.to(dtype=torch.float16)
            latents = latents.to(dtype=torch.float16)
            target_latents = target_latents.to(dtype=torch.float16)
            
            pred = self.unet(
                sample=latents,
                timestep=timestep,
                encoder_hidden_states=text_embeddings
            ).sample
            
            # Ensure loss computation is in float16
            loss = criterion(pred.to(dtype=torch.float16), target_latents)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            lrs.append(lr)
            losses.append(loss.item())
            
            lr *= lr_factor
            
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
            
            if loss.item() > 4 * self.best_loss:
                break
                
        self.history["lr"] = lrs
        self.history["loss"] = losses

    def plot(self, skip_start=0, skip_end=0):
        lrs = self.history["lr"][skip_start:-skip_end] if skip_end > 0 else self.history["lr"][skip_start:]
        losses = self.history["loss"][skip_start:-skip_end] if skip_end > 0 else self.history["loss"][skip_start:]

        # Check if we have enough points
        if len(losses) < 2:
            print("Not enough points to plot. Try increasing num_iter or reducing skip_start/skip_end")
            return None
        
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate vs Loss')
        plt.grid(True)
        plt.show()

        # Find learning rate with steepest gradient
        if len(losses) >= 4:  # Need at least 4 points for reliable gradient calculation
            min_grad_idx = (torch.gradient(torch.tensor(losses))[0]).argmin()
            suggested_lr = lrs[min_grad_idx]
        else:
            # Fallback to simpler method if we have too few points
            min_loss_idx = losses.index(min(losses))
            suggested_lr = lrs[min_loss_idx]
            
        print(f"Suggested learning rate: {suggested_lr:.2e}")
        return suggested_lr

def main():
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Learning rate finder for Stable Diffusion 1.5')
    parser.add_argument('--model_path', type=str, default="SD", 
                        help='Path to SD model directory')
    parser.add_argument('--dataset_path', type=str, default="dataset",
                        help='Path to dataset directory')
    parser.add_argument('--start_lr', type=float, default=2e-4,
                        help='Starting learning rate')
    parser.add_argument('--end_lr', type=float, default=2e-1,
                        help='Final learning rate')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--num_iter', type=int, default=83,
                        help='Number of iterations')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # Фиксируем seed для воспроизводимости
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Параметры из аргументов
    MODEL_PATH = args.model_path
    DATASET_PATH = args.dataset_path
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = args.batch_size

    try:
        # Проверяем наличие файлов
        model_path = Path(MODEL_PATH)
        required_files = [
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "text_encoder/model.fp16.safetensors",
            "text_encoder/config.json"
        ]
        
        for file in required_files:
            if not (model_path / file).exists():
                raise ValueError(f"Required file not found: {model_path / file}")
        
        # Подготовка данных
        dataset = SDDataset(DATASET_PATH)
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")
            
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Создаем и загружаем модели напрямую через SDLRFinder
        finder = SDLRFinder(MODEL_PATH, DEVICE)
        
        # Оптимизатор и функция потерь с поддержкой fp16
        optimizer = torch.optim.AdamW(finder.unet.parameters(), eps=1e-4)
        criterion = torch.nn.HuberLoss().to(DEVICE, dtype=torch.float16)
        
        print("\nStarting learning rate search...")
        print(f"Parameters:")
        print(f"- Start LR: {args.start_lr}")
        print(f"- End LR: {args.end_lr}")
        print(f"- Batch size: {args.batch_size}")
        print(f"- Iterations: {args.num_iter}")
        
        finder.find_lr(
            dataloader,
            optimizer,
            criterion,
            min_lr=args.start_lr,
            max_lr=args.end_lr,
            num_iter=args.num_iter
        )
        
        print("\nPlotting results...")
        suggested_lr = finder.plot()
        
        # Сохранение результата
        with open("suggested_lr.txt", "w") as f:
            f.write(f"Suggested learning rate: {suggested_lr:.2e}")
        
        print(f"\nSearch completed! Results saved to suggested_lr.txt")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease ensure:")
        print("1. The 'dataset' folder exists and contains images")
        print("2. The 'SD' folder contains the model files")
        print("3. You have proper permissions to access these folders")
        return

if __name__ == "__main__":
    main()
