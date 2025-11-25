"""
모델 학습 스크립트
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from model import ReactionPredictor
from data_processor import prepare_dataset
import matplotlib.pyplot as plt


def load_data(filepath: str = 'reaction_data.csv'):
    """CSV 파일에서 반응 데이터 로드"""
    df = pd.read_csv(filepath)
    reactions = list(zip(df['reactant'], df['product'], df['success_prob']))
    return reactions


def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cpu'):
    """모델 학습"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses


def plot_training_history(train_losses, val_losses):
    """학습 과정 시각화"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')
    print("학습 그래프가 'training_history.png'로 저장되었습니다.")


def main():
    # 하이퍼파라미터
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"사용 디바이스: {DEVICE}")
    
    # 데이터 로드
    print("데이터 로드 중...")
    reactions = load_data('reaction_data.csv')
    X, y = prepare_dataset(reactions)
    
    print(f"데이터 크기: {X.shape}, 레이블 크기: {y.shape}")
    
    # Train/Val 분할
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # PyTorch 데이터셋 생성
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 모델 생성
    input_size = X.shape[1]
    model = ReactionPredictor(input_size=input_size)
    print(f"모델 생성 완료 (입력 크기: {input_size})")
    
    # 학습
    print("학습 시작...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE
    )
    
    # 모델 저장
    torch.save(model.state_dict(), 'reaction_predictor.pth')
    print("모델이 'reaction_predictor.pth'로 저장되었습니다.")
    
    # 학습 과정 시각화
    plot_training_history(train_losses, val_losses)


if __name__ == '__main__':
    main()
