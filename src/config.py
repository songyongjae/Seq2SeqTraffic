import torch
from sklearn.preprocessing import MinMaxScaler
from model import lstm_encoder_decoder
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else 'cpu'
MODEL = lstm_encoder_decoder(input_size=1, hidden_size=16).to(DEVICE)
CRITERION = nn.MSELoss()
SCALER = MinMaxScaler()
LEARNING_RATE = 0.015
EPOCHS = 700
OPTIMIZER = optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)

QUERY = [
    '쌀', '콩', '라면', '고구마', '버섯', '토마토', '양파', '사과', '아보카도', '두부',
    '월세', '도시가스', '전기료', '소형승용차', '휘발유', '주차료', '자동차보험료',
    '보험서비스료', '온라인콘텐츠이용료', '휴대전화료', '인터넷이용료', '운동강습료',
    '반려동물관리비', '반려동물용품', '컴퓨터소모품', '부엌용세제', '담배', '화장지',
    '이발료', '남자상의'
]
