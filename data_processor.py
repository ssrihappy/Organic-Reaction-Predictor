"""
SMILES 데이터 전처리 모듈
분자 구조를 벡터로 변환하고 반응성 작용기를 추출
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from typing import List, Tuple, Optional


class SMILESProcessor:
    """SMILES 구조를 처리하고 특징 벡터로 변환"""
    
    def __init__(self, fp_size: int = 2048, radius: int = 2):
        self.fp_size = fp_size
        self.radius = radius
        
    def smiles_to_fingerprint(self, smiles: str) -> Optional[np.ndarray]:
        """SMILES를 Morgan Fingerprint로 변환"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.fp_size)
            return np.array(fp)
        except:
            return None
    
    def extract_functional_groups(self, smiles: str) -> dict:
        """반응성 작용기 추출"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        # Br2와 반응 가능한 주요 작용기 패턴
        functional_groups = {
            'alkene': len(mol.GetSubstructMatches(Chem.MolFromSmarts('C=C'))),
            'alkyne': len(mol.GetSubstructMatches(Chem.MolFromSmarts('C#C'))),
            'aromatic': len(mol.GetSubstructMatches(Chem.MolFromSmarts('c'))),
            'alcohol': len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OH]'))),
            'amine': len(mol.GetSubstructMatches(Chem.MolFromSmarts('[NH2,NH1,NH0]'))),
        }
        return functional_groups
    
    def get_molecular_descriptors(self, smiles: str) -> Optional[np.ndarray]:
        """분자 기술자 추출"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        descriptors = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
        ]
        return np.array(descriptors)
    
    def process_reaction(self, reactant_smiles: str, product_smiles: str) -> Optional[np.ndarray]:
        """반응물과 생성물을 처리하여 특징 벡터 생성"""
        reactant_fp = self.smiles_to_fingerprint(reactant_smiles)
        product_fp = self.smiles_to_fingerprint(product_smiles)
        reactant_desc = self.get_molecular_descriptors(reactant_smiles)
        
        if reactant_fp is None or product_fp is None or reactant_desc is None:
            return None
        
        # 반응물, 생성물, 기술자를 결합
        features = np.concatenate([reactant_fp, product_fp, reactant_desc])
        return features


def prepare_dataset(reactions: List[Tuple[str, str, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    반응 데이터셋 준비
    
    Args:
        reactions: [(reactant_smiles, product_smiles, success_probability), ...]
    
    Returns:
        X: 특징 행렬, y: 성공 확률 레이블
    """
    processor = SMILESProcessor()
    X_list = []
    y_list = []
    
    for reactant, product, success_prob in reactions:
        features = processor.process_reaction(reactant, product)
        if features is not None:
            X_list.append(features)
            y_list.append(success_prob)
    
    return np.array(X_list), np.array(y_list)
