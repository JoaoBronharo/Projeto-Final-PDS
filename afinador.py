import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

# ===========================================================
# CONFIGURAÇÕES DO SISTEMA
# ===========================================================
# SR: taxa de amostragem para a qual todos os áudios serão convertidos
# FRAME_LENGTH e HOP_LENGTH: parâmetros usados pelo algoritmo YIN de detecção de f0
# FMIN e FMAX: faixa de busca para a frequência fundamental
# IGNORE_ATTACK_SECONDS: tempo inicial removido devido à instabilidade do ataque da nota
# SMOOTH_WINDOW: tamanho da janela para suavizar a curva final de f0

SR = 48000
FRAME_LENGTH = 2048
HOP_LENGTH = 1024

FMIN = 50
FMAX = 400

IGNORE_ATTACK_SECONDS = 0.05
SMOOTH_WINDOW = 5


# ===========================================================
# SUAVIZAÇÃO SEGURA
# ===========================================================
# moving_average(): aplica uma média móvel simples em um vetor de f0,
# garantindo suavização sem alterar a estrutura global do sinal

def moving_average(x, w):
    x = np.array(x).flatten()
    if len(x) < w:
        return x  # evita erro caso o vetor seja muito curto
    return np.convolve(x, np.ones(w) / w, mode='same')


# ===========================================================
# CONVERSÃO DE ERRO PARA CENTS
# ===========================================================
# converte o erro entre frequência detectada e frequência ideal para cents
# 1200 * log2(f_medida / f_ref) é a fórmula padrão da escala musical

def erro_em_cents(f_medida, f_ref):
    return 1200 * np.log2(f_medida / f_ref)


# ===========================================================
# PROCESSAMENTO DE CADA ÁUDIO
# ===========================================================
# esta função faz todo o pipeline: leitura do áudio, remoção do ataque,
# resample, detecção de f0, FFT comparativa e geração das figuras

def processar_audio(filepath, freq_ideal, nome):

    print(f"\nProcessando {nome} ...")

    # -----------------------------
    # CARREGAR ÁUDIO SEM LIBROSA.LOAD
    # -----------------------------
    # wavfile.read lê o áudio mantendo a amostragem original do arquivo
    fs, audio = wavfile.read(filepath)

    # caso o áudio tenha dois canais, usa apenas o canal esquerdo
    if audio.ndim > 1:
        audio = audio[:, 0]

    # normaliza amplitude para valores entre -1 e 1
    audio = audio.astype(np.float32)
    audio /= np.max(np.abs(audio))

    # remove o ataque inicial instável da corda
    inicio = int(IGNORE_ATTACK_SECONDS * fs)
    audio = audio[inicio:]

    # garante que o áudio tenha tamanho mínimo para FFT
    if len(audio) < 4096:
        print("⚠ Áudio curto após corte do ataque. Ajustando...")
        audio = np.pad(audio, (0, 4096 - len(audio)))

    # reamostragem do áudio para 48 kHz
    audio = librosa.resample(audio, orig_sr=fs, target_sr=SR)
    fs = SR

    # ===========================================================
    # DETECÇÃO DE F0 — YIN
    # ===========================================================
    # o algoritmo YIN detecta a frequência fundamental em cada janela temporal

    f0 = librosa.yin(
        audio,
        fmin=FMIN,
        fmax=FMAX,
        sr=fs,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        trough_threshold=0.1
    )

    # remove valores fora da faixa de interesse
    f0[(f0 < FMIN) | (f0 > FMAX)] = np.nan

    # suavização da curva de f0
    f0_smooth = moving_average(f0, SMOOTH_WINDOW)

    # interpola valores faltantes (NaN)
    valid = ~np.isnan(f0_smooth)
    if np.sum(valid) > 1:
        f0_smooth = np.interp(
            np.arange(len(f0_smooth)),
            np.where(valid)[0],
            f0_smooth[valid]
        )

    # ===========================================================
    # ERRO EM CENTS
    # ===========================================================
    # converte f0 estimada em erro tonal perceptivo
    erro_cents = erro_em_cents(f0_smooth, freq_ideal)

    # ===========================================================
    # FFT — TAMANHOS DIFERENTES
    # ===========================================================
    # gera FFT com tamanhos diferentes para comparar resolução espectral

    def gerar_fft(audio, fs, N):
        janela = np.hanning(N)
        fft = np.abs(np.fft.rfft(audio[:N] * janela))  # usa apenas a parte positiva
        freqs = np.fft.rfftfreq(N, 1/fs)
        return freqs, fft

    freqs_2048, fft_2048 = gerar_fft(audio, fs, 2048)
    freqs_8192, fft_8192 = gerar_fft(audio, fs, 8192)

    # ===========================================================
    # Comparação de janelas (Hann vs Hamming)
    # ===========================================================

    def gerar_fft_janela(audio, fs, N, janela):
        win = janela(N)
        fft = np.abs(np.fft.rfft(audio[:N] * win))
        freqs = np.fft.rfftfreq(N, 1/fs)
        return freqs, fft

    freqs_hann, fft_hann = gerar_fft_janela(audio, fs, 4096, np.hanning)
    freqs_hamm, fft_hamm = gerar_fft_janela(audio, fs, 4096, np.hamming)

    # ===========================================================
    # criar pastas
    # ===========================================================
    # garante que a pasta de figuras exista
    os.makedirs("figuras_relatorio", exist_ok=True)

    # ===========================================================
    # GRÁFICOS
    # ===========================================================

    # ----------- F0 vs tempo ------------
    plt.figure(figsize=(12, 5))
    plt.plot(f0_smooth, label="f0 estimada (YIN)")
    plt.axhline(freq_ideal, color="red", linestyle="--", label=f"Ideal = {freq_ideal} Hz")
    plt.title(f"Frequência Fundamental — Nota {nome}")
    plt.xlabel("Janela")
    plt.ylabel("Frequência (Hz)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figuras_relatorio/f0_{nome}.png")
    plt.close()

    # -------- HISTOGRAMA DO ERRO EM CENTS --------
    plt.figure(figsize=(10, 5))
    plt.hist(erro_cents, bins=50, color="purple", alpha=0.7)
    plt.axvline(0, color="black", linestyle="--")
    plt.title(f"Histograma do Erro em Cents — Nota {nome}")
    plt.xlabel("Erro (cents)")
    plt.ylabel("Ocorrências")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figuras_relatorio/erro_cents_{nome}.png")
    plt.close()

    # -------- FFT COMPARATIVA 2048 vs 8192 --------
    plt.figure(figsize=(12, 5))
    plt.plot(freqs_2048, fft_2048, label="FFT 2048")
    plt.plot(freqs_8192, fft_8192, label="FFT 8192", alpha=0.7)
    plt.xlim(0, 2000)
    plt.title(f"FFT (Comparação N=2048 vs N=8192) — {nome}")
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figuras_relatorio/fft_comp_{nome}.png")
    plt.close()

    # -------- FFT Hann vs Hamming --------
    plt.figure(figsize=(12, 5))
    plt.plot(freqs_hann, fft_hann, label="Hann")
    plt.plot(freqs_hamm, fft_hamm, label="Hamming")
    plt.xlim(0, 2000)
    plt.title(f"FFT com Janelas Diferentes — {nome}")
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figuras_relatorio/fft_janelas_{nome}.png")
    plt.close()

    print(f"✓ Gráficos gerados para {nome}")


# ===========================================================
# ARQUIVOS
# ===========================================================
# cicionário contendo o nome da nota, o arquivo correspondente
# e a frequência ideal usada para cálculo do erro

notas = {
    "A2": ("nota_A2.wav", 110),
    "D3": ("nota_D3.wav", 147),
    "G3": ("nota_G3.wav", 196),
    "B3": ("nota_B3.wav", 247)
}

# loop que processa cada nota
for nome, (arquivo, freq) in notas.items():
    processar_audio(arquivo, freq, nome)

print("\n✔ PROCESSO FINALIZADO!")
print("Imagens salvas em: figuras_relatorio/")
