"""
학습된 모델로 반응 성공률 예측
"""
import torch
import numpy as np
from model import ReactionPredictor
from data_processor import SMILESProcessor


class ReactionSuccessPredictor:
    """반응 성공률 예측기"""
    
    def __init__(self, model_path: str = 'reaction_predictor.pth', input_size: int = 4102):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ReactionPredictor(input_size=input_size)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.processor = SMILESProcessor()
    
    def predict(self, reactant_smiles: str, product_smiles: str, reagent: str = 'Br2') -> float:
        """
        반응 성공 확률 예측
        
        Args:
            reactant_smiles: 반응물 SMILES
            product_smiles: 생성물 SMILES
            reagent: 시약 (현재 MVP는 Br2로 고정)
        
        Returns:
            성공 확률 (0~1)
        """
        # 특징 추출
        features = self.processor.process_reaction(reactant_smiles, product_smiles)
        
        if features is None:
            raise ValueError("유효하지 않은 SMILES 구조입니다.")
        
        # 예측
        with torch.no_grad():
            X = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            probability = self.model(X).item()
        
        return probability
    
    def analyze_functional_groups(self, reactant_smiles: str):
        """반응물의 작용기 분석"""
        groups = self.processor.extract_functional_groups(reactant_smiles)
        print(f"\n=== 작용기 분석 (시약: Br2) ===")
        print(f"Alkene (C=C): {groups.get('alkene', 0)}개")
        print(f"Alkyne (C≡C): {groups.get('alkyne', 0)}개")
        print(f"Aromatic: {groups.get('aromatic', 0)}개")
        print(f"Alcohol (-OH): {groups.get('alcohol', 0)}개")
        print(f"Amine (-NH2): {groups.get('amine', 0)}개")
        return groups


def main():
    """예측 예제"""
    predictor = ReactionSuccessPredictor()
    
    # 예제 반응들
    examples = [
        {
            'name': 'Ethene + Br2 (알켄 브롬화)',
            'reactant': 'C=C',
            'product': 'C(Br)C(Br)'
        },
        {
            'name': 'Benzene + Br2 (방향족 브롬화)',
            'reactant': 'c1ccccc1',
            'product': 'c1ccc(Br)cc1'
        },
        {
            'name': 'Propene + Br2',
            'reactant': 'CC=C',
            'product': 'CC(Br)C(Br)'
        }
    ]
    
    print("=" * 60)
    print("유기화학 반응 성공률 예측 시스템 (MVP - Br2 시약)")
    print("=" * 60)
    
    for example in examples:
        print(f"\n반응: {example['name']}")
        print(f"반응물: {example['reactant']}")
        print(f"생성물: {example['product']}")
        
        try:
            # 작용기 분석
            predictor.analyze_functional_groups(example['reactant'])
            
            # 성공률 예측
            probability = predictor.predict(example['reactant'], example['product'])
            print(f"\n예측 성공률: {probability*100:.2f}%")
            
            if probability > 0.7:
                print("판정: 반응 성공 가능성 높음 ✓")
            elif probability > 0.4:
                print("판정: 반응 성공 가능성 보통")
            else:
                print("판정: 반응 성공 가능성 낮음 ✗")
                
        except Exception as e:
            print(f"오류: {e}")
        
        print("-" * 60)


if __name__ == '__main__':
    main()
