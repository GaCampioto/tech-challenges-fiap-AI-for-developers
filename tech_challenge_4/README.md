# Tech Challenge 4

## O problema
O Tech Challenge desta fase será a criação de uma aplicação que utilize
análise de vídeo. O seu projeto deve incorporar as técnicas de reconhecimento
facial, análise de expressões emocionais em vídeos e detecção de atividades.

## A proposta do desafio
Você deverá criar uma aplicação a partir do vídeo que se encontra
disponível na plataforma do aluno, e que execute as seguintes tarefas:

## Implementação
O desafio foi implementado em python 3.11, utilizando as bibliotecas:
- opencv
- mediapipe
- deepface
- tqdm

Não achei necessário realizar a sumarização automática, uma vez que o código já organiza e gera um arquivo de saída com as informações necessárias.

## Execução
`pip install opencv-python mediapipe deepface tqdm tf_keras`

`python emotion_analysis.py`