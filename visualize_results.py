"""
브롬화 반응 예측 결과 시각화 및 요약
"""
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
import os


def visualize_bromination_results():
    """브롬화 결과 시각화"""
    
    print("=" * 80)
    print("브롬화 반응 예측 결과 요약")
    print("=" * 80)
    
    # 원본 구조 로드
    try:
        from target_smiles import target
        target_smiles = target
    except:
        print("오류: target_smiles.py를 찾을 수 없습니다.")
        return
    
    # 생성물 데이터 로드
    if not os.path.exists('brominated_products_smiles.csv'):
        print("오류: brominated_products_smiles.csv를 찾을 수 없습니다.")
        print("먼저 predict_target.py를 실행하세요.")
        return
    
    df = pd.read_csv('brominated_products_smiles.csv')
    
    print("\n[원본 구조]")
    print("-" * 80)
    print(f"SMILES: {target_smiles[:100]}...")
    
    mol = Chem.MolFromSmiles(target_smiles)
    if mol:
        print(f"분자식: {Chem.rdMolDescriptors.CalcMolFormula(mol)}")
        print(f"분자량: {Descriptors.MolWt(mol):.2f} g/mol")
    
    print("\n[예측된 브롬화 생성물]")
    print("-" * 80)
    
    for idx, row in df.iterrows():
        print(f"\n생성물 #{row['pathway_number']}")
        print(f"  반응 타입: {row['reaction_type']}")
        print(f"  브롬화 위치: {row['bromination_site']}")
        print(f"  성공 확률: {row['success_probability']*100:.1f}%")
        print(f"  예상 수율: {row['expected_yield']*100:.1f}%")
        print(f"\n  생성물 SMILES:")
        print(f"  {row['product_smiles']}")
        
        # 생성물 분자 정보
        product_mol = Chem.MolFromSmiles(row['product_smiles'])
        if product_mol:
            print(f"\n  생성물 분자식: {Chem.rdMolDescriptors.CalcMolFormula(product_mol)}")
            print(f"  생성물 분자량: {Descriptors.MolWt(product_mol):.2f} g/mol")
            
            # Br 원자 개수 확인
            br_count = sum(1 for atom in product_mol.GetAtoms() if atom.GetSymbol() == 'Br')
            print(f"  Br 원자 개수: {br_count}개")
    
    print("\n" + "=" * 80)
    print("결과 파일:")
    print("  - brominated_products_smiles.csv: 생성물 SMILES 및 예측 정보")
    print("  - target_prediction_results.csv: 전체 예측 요약")
    print("  - target_reaction_pathways.csv: 반응 경로 상세")
    print("=" * 80)
    
    # 구조 이미지 생성 시도
    try:
        print("\n[구조 이미지 생성 시도]")
        
        # 원본 구조
        if mol:
            img = Draw.MolToImage(mol, size=(800, 600))
            img.save('target_structure.png')
            print("✓ 원본 구조: target_structure.png")
        
        # 생성물 구조들
        for idx, row in df.iterrows():
            product_mol = Chem.MolFromSmiles(row['product_smiles'])
            if product_mol:
                # Br 원자 하이라이트
                br_atoms = [atom.GetIdx() for atom in product_mol.GetAtoms() if atom.GetSymbol() == 'Br']
                
                img = Draw.MolToImage(product_mol, size=(800, 600), highlightAtoms=br_atoms)
                img.save(f'brominated_product_{row["pathway_number"]}.png')
                print(f"✓ 생성물 #{row['pathway_number']}: brominated_product_{row['pathway_number']}.png")
        
        print("\n구조 이미지가 생성되었습니다. (Br 원자가 하이라이트됨)")
        
    except Exception as e:
        print(f"\n⚠ 이미지 생성 실패: {e}")
        print("이미지 생성을 위해서는 pillow 패키지가 필요합니다: pip install pillow")


def compare_structures():
    """원본과 생성물 구조 비교"""
    print("\n" + "=" * 80)
    print("구조 비교 분석")
    print("=" * 80)
    
    try:
        from target_smiles import target
        target_smiles = target
    except:
        return
    
    if not os.path.exists('brominated_products_smiles.csv'):
        return
    
    df = pd.read_csv('brominated_products_smiles.csv')
    
    original_mol = Chem.MolFromSmiles(target_smiles)
    
    if not original_mol:
        return
    
    original_mw = Descriptors.MolWt(original_mol)
    
    print(f"\n원본 분자량: {original_mw:.2f} g/mol")
    
    for idx, row in df.iterrows():
        product_mol = Chem.MolFromSmiles(row['product_smiles'])
        if product_mol:
            product_mw = Descriptors.MolWt(product_mol)
            mw_increase = product_mw - original_mw
            
            print(f"\n생성물 #{row['pathway_number']}:")
            print(f"  분자량: {product_mw:.2f} g/mol")
            print(f"  증가량: +{mw_increase:.2f} g/mol")
            print(f"  (Br 1개 = ~80 g/mol)")


if __name__ == '__main__':
    visualize_bromination_results()
    compare_structures()
