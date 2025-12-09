# 🎼 Afinador Digital – Processamento Digital de Sinais (UTFPR)

> Projeto final desenvolvido na disciplina **Processamento Digital de Sinais – 2025/2**,  
> explorando técnicas de DSP para análise espectral, estimação de frequência fundamental (f₀)  
> e avaliação do erro tonal (cents) em notas musicais gravadas de um violão.

## 👥 Integrantes do Projeto

| Nome | RA |
|------|------|
| **João Pedro Garcia Bronharo** | 2553066 |
| **Caio Vinícius Maciel Delgado** | 2552949 |

## 📁 Estrutura do Repositório

```
afinador-digital-PDS/
│
├── afinador.py
├── nota_A2.wav
├── nota_B3.wav
├── nota_D3.wav
├── nota_G3.wav
│
└── figuras_relatorio/
       ├── f0_A2.png
       ├── hist_A2.png
       ├── fft_A2_2048_8192.png
       ├── fft_A2_janelas.png
       │
       ├── f0_B3.png
       ├── hist_B3.png
       ├── fft_B3_2048_8192.png
       ├── fft_B3_janelas.png
       │
       ├── f0_D3.png
       ├── hist_D3.png
       ├── fft_D3_2048_8192.png
       ├── fft_D3_janelas.png
       │
       ├── f0_G3.png
       ├── hist_G3.png
       ├── fft_G3_2048_8192.png
       ├── fft_G3_janelas.png

Total: 16 figuras (4 por nota).
```

## 🎯 Objetivo do Projeto

Criar um **afinador digital** capaz de:

- extrair a frequência fundamental (**f₀**) de notas musicais reais,  
- analisar espectros via FFT,  
- comparar janelas (Hann × Hamming),  
- estudar a influência da resolução espectral (2048 × 8192 pontos),  
- calcular o erro tonal em **cents**,  
- e avaliar a estabilidade temporal da nota através do algoritmo **YIN**.

## 🛠️ Tecnologias e Bibliotecas Utilizadas

- **Python 3.11**
- **NumPy**
- **SciPy**
- **Librosa**
- **Matplotlib**
- **SoundFile**

## ▶️ Como Executar o Afinador

1. Instale as dependências:

```bash
pip install numpy scipy librosa matplotlib soundfile
```

2. Clone o repositório:

```bash
git clone https://github.com/SEU-USUARIO/afinador-digital-PDS
cd afinador-digital-PDS
```

3. Execute o script:

```bash
python afinador.py
```

As figuras serão geradas em:

```
figuras_relatorio/
```

## 🧠 Pipeline de Processamento do Sinal

1) **Leitura e normalização**  
2) **Remoção do ataque**  
3) **Reamostragem para 48 kHz**  
4) **FFT + comparação de janelas e tamanhos**  
5) **Estimativa de f₀ via YIN (50–400 Hz)**  
6) **Cálculo do erro em cents:**  

```
epsilon = 1200 * log2(f0 / f_ideal)
```

7) **Geração das figuras**

## 📈 Resultados Obtidos

- FFT de 8192 pontos → melhor definição harmônica  
- Janela Hann → menor leakage  
- YIN → estável após remoção do ataque  
- Erros em cents próximos do ideal  

## 🔬 Reprodutibilidade

Execute:

```bash
python afinador.py
```

## 📝 Melhorias Futuras

- Afinador em tempo real  
- Interface gráfica  
- Afinador cromático  
- Filtros de redução de ruído  

## 📚 Referências

- https://librosa.org  
- Documentação NumPy FFT  
- Material da disciplina de PDS — UTFPR  