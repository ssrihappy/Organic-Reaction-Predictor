"""
샘플 학습 데이터 생성
실제 사용 시에는 실험 데이터베이스에서 가져와야 함
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_realistic_reactions():
    """Br2 시약을 사용한 현실적인 반응 데이터 생성 (yield 포함)"""
    
    # 반응 템플릿: (반응물, 생성물, 평균 yield, yield 표준편차, 반응 조건)
    reaction_templates = [
        # === 알켄 브롬화 (높은 수율) ===
        {
            'reactant': 'C=C',
            'product': 'C(Br)C(Br)',
            'name': 'Ethene bromination',
            'avg_yield': 0.92,
            'std_yield': 0.05,
            'temp': 25,
            'solvent': 'CCl4',
            'time': 2.0
        },
        {
            'reactant': 'CC=C',
            'product': 'CC(Br)C(Br)',
            'name': 'Propene bromination',
            'avg_yield': 0.88,
            'std_yield': 0.06,
            'temp': 20,
            'solvent': 'CCl4',
            'time': 2.5
        },
        {
            'reactant': 'CC=CC',
            'product': 'CC(Br)C(Br)C',
            'name': '2-Butene bromination',
            'avg_yield': 0.85,
            'std_yield': 0.07,
            'temp': 25,
            'solvent': 'CH2Cl2',
            'time': 3.0
        },
        {
            'reactant': 'C=CC=C',
            'product': 'C(Br)CC(Br)C',
            'name': '1,3-Butadiene bromination',
            'avg_yield': 0.80,
            'std_yield': 0.08,
            'temp': 0,
            'solvent': 'CCl4',
            'time': 4.0
        },
        {
            'reactant': 'C1=CCCCC1',
            'product': 'C1C(Br)C(Br)CCC1',
            'name': 'Cyclohexene bromination',
            'avg_yield': 0.83,
            'std_yield': 0.06,
            'temp': 25,
            'solvent': 'CCl4',
            'time': 3.5
        },
        {
            'reactant': 'CC(C)=C',
            'product': 'CC(C)(Br)C(Br)',
            'name': 'Isobutene bromination',
            'avg_yield': 0.86,
            'std_yield': 0.06,
            'temp': 20,
            'solvent': 'CH2Cl2',
            'time': 2.5
        },
        {
            'reactant': 'C=CC(C)C',
            'product': 'C(Br)C(Br)C(C)C',
            'name': '3-Methyl-1-butene bromination',
            'avg_yield': 0.84,
            'std_yield': 0.07,
            'temp': 25,
            'solvent': 'CCl4',
            'time': 3.0
        },
        
        # === 알카인 브롬화 (중간 수율) ===
        {
            'reactant': 'C#C',
            'product': 'C(Br)=C(Br)',
            'name': 'Acetylene bromination',
            'avg_yield': 0.75,
            'std_yield': 0.08,
            'temp': 0,
            'solvent': 'CCl4',
            'time': 5.0
        },
        {
            'reactant': 'CC#C',
            'product': 'CC(Br)=C(Br)',
            'name': 'Propyne bromination',
            'avg_yield': 0.72,
            'std_yield': 0.09,
            'temp': 0,
            'solvent': 'CCl4',
            'time': 5.5
        },
        {
            'reactant': 'CC#CC',
            'product': 'CC(Br)=C(Br)C',
            'name': '2-Butyne bromination',
            'avg_yield': 0.70,
            'std_yield': 0.10,
            'temp': 5,
            'solvent': 'CH2Cl2',
            'time': 6.0
        },
        
        # === 방향족 브롬화 (중간 수율, 촉매 필요) ===
        {
            'reactant': 'c1ccccc1',
            'product': 'c1ccc(Br)cc1',
            'name': 'Benzene bromination',
            'avg_yield': 0.65,
            'std_yield': 0.10,
            'temp': 25,
            'solvent': 'none',
            'time': 8.0
        },
        {
            'reactant': 'c1ccc(C)cc1',
            'product': 'c1ccc(C)c(Br)c1',
            'name': 'Toluene bromination',
            'avg_yield': 0.70,
            'std_yield': 0.09,
            'temp': 25,
            'solvent': 'none',
            'time': 6.0
        },
        {
            'reactant': 'c1ccc(O)cc1',
            'product': 'c1ccc(O)c(Br)c1',
            'name': 'Phenol bromination',
            'avg_yield': 0.78,
            'std_yield': 0.08,
            'temp': 20,
            'solvent': 'H2O',
            'time': 4.0
        },
        {
            'reactant': 'c1ccc(N)cc1',
            'product': 'c1ccc(N)c(Br)c1',
            'name': 'Aniline bromination',
            'avg_yield': 0.75,
            'std_yield': 0.08,
            'temp': 20,
            'solvent': 'H2O',
            'time': 4.5
        },
        {
            'reactant': 'c1ccc(OC)cc1',
            'product': 'c1ccc(OC)c(Br)c1',
            'name': 'Anisole bromination',
            'avg_yield': 0.73,
            'std_yield': 0.09,
            'temp': 25,
            'solvent': 'CCl4',
            'time': 7.0
        },
        
        # === 공액 시스템 ===
        {
            'reactant': 'c1ccccc1C=C',
            'product': 'c1ccccc1C(Br)C(Br)',
            'name': 'Styrene bromination',
            'avg_yield': 0.82,
            'std_yield': 0.07,
            'temp': 25,
            'solvent': 'CCl4',
            'time': 3.5
        },
        {
            'reactant': 'CC(=O)C=C',
            'product': 'CC(=O)C(Br)C(Br)',
            'name': 'Methyl vinyl ketone bromination',
            'avg_yield': 0.68,
            'std_yield': 0.10,
            'temp': 20,
            'solvent': 'CH2Cl2',
            'time': 4.0
        },
        
        # === 알칸 (낮은 수율 - 라디칼 반응) ===
        {
            'reactant': 'CC',
            'product': 'C(Br)C',
            'name': 'Ethane bromination',
            'avg_yield': 0.25,
            'std_yield': 0.08,
            'temp': 400,
            'solvent': 'none',
            'time': 12.0
        },
        {
            'reactant': 'CCC',
            'product': 'CC(Br)C',
            'name': 'Propane bromination',
            'avg_yield': 0.22,
            'std_yield': 0.09,
            'temp': 400,
            'solvent': 'none',
            'time': 12.0
        },
        {
            'reactant': 'CCCC',
            'product': 'CCC(Br)C',
            'name': 'Butane bromination',
            'avg_yield': 0.20,
            'std_yield': 0.10,
            'temp': 400,
            'solvent': 'none',
            'time': 12.0
        },
        {
            'reactant': 'CC(C)C',
            'product': 'CC(C)(Br)C',
            'name': 'Isobutane bromination',
            'avg_yield': 0.28,
            'std_yield': 0.08,
            'temp': 400,
            'solvent': 'none',
            'time': 10.0
        },
    ]
    
    # 각 반응에 대해 여러 실험 데이터 생성
    all_reactions = []
    reaction_id = 1
    base_date = datetime(2023, 1, 1)
    
    for template in reaction_templates:
        # 각 반응 타입당 15-25개의 실험 데이터 생성
        num_experiments = np.random.randint(15, 26)
        
        for i in range(num_experiments):
            # yield에 정규분포 노이즈 추가
            yield_value = np.random.normal(template['avg_yield'], template['std_yield'])
            yield_value = np.clip(yield_value, 0.0, 1.0)
            
            # 반응 조건에 약간의 변동 추가
            temp_variation = np.random.uniform(-5, 5)
            time_variation = np.random.uniform(0.8, 1.2)
            
            # 실험 날짜 (랜덤)
            days_offset = np.random.randint(0, 730)  # 2년 범위
            experiment_date = base_date + timedelta(days=days_offset)
            
            # 성공 확률 계산 (yield 기반)
            # yield가 높을수록 성공 확률도 높음
            success_prob = yield_value * np.random.uniform(0.95, 1.05)
            success_prob = np.clip(success_prob, 0.0, 1.0)
            
            reaction_data = {
                'reaction_id': f'RXN{reaction_id:04d}',
                'reaction_name': template['name'],
                'reactant': template['reactant'],
                'product': template['product'],
                'reagent': 'Br2',
                'yield': round(yield_value, 4),
                'success_prob': round(success_prob, 4),
                'temperature': round(template['temp'] + temp_variation, 1),
                'solvent': template['solvent'],
                'reaction_time': round(template['time'] * time_variation, 2),
                'experiment_date': experiment_date.strftime('%Y-%m-%d'),
                'batch_size': np.random.choice([10, 25, 50, 100]),  # mmol
            }
            
            all_reactions.append(reaction_data)
            reaction_id += 1
    
    return all_reactions


def main():
    """샘플 데이터 생성 및 저장"""
    print("=" * 70)
    print("현실적인 유기화학 반응 데이터 생성 중...")
    print("=" * 70)
    
    np.random.seed(42)  # 재현성을 위한 시드 설정
    
    reactions = generate_realistic_reactions()
    df = pd.DataFrame(reactions)
    
    # CSV로 저장
    df.to_csv('reaction_data.csv', index=False)
    
    print(f"\n✓ 총 {len(reactions)}개의 반응 데이터가 생성되었습니다.")
    print(f"✓ 파일: reaction_data.csv")
    
    print("\n" + "=" * 70)
    print("데이터 통계")
    print("=" * 70)
    
    print(f"\n고유 반응 타입: {df['reaction_name'].nunique()}개")
    print(f"실험 기간: {df['experiment_date'].min()} ~ {df['experiment_date'].max()}")
    
    print("\n[Yield 분포]")
    print(df['yield'].describe())
    
    print("\n[성공 확률 분포]")
    print(df['success_prob'].describe())
    
    print("\n[반응 타입별 평균 Yield]")
    reaction_summary = df.groupby('reaction_name').agg({
        'yield': ['mean', 'std', 'count'],
        'success_prob': 'mean'
    }).round(4)
    reaction_summary.columns = ['Avg_Yield', 'Std_Yield', 'Count', 'Avg_Success_Prob']
    reaction_summary = reaction_summary.sort_values('Avg_Yield', ascending=False)
    print(reaction_summary.head(10))
    
    print("\n[데이터 샘플 (처음 10개)]")
    print(df[['reaction_id', 'reaction_name', 'reactant', 'product', 'yield', 'success_prob', 'temperature']].head(10))
    
    print("\n" + "=" * 70)
    print("데이터 생성 완료!")
    print("=" * 70)


if __name__ == '__main__':
    main()
