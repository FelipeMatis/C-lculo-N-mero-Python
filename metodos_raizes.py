# ===============================================================
# Métodos Numéricos para Encontrar Raízes:
# Bisseção, Ponto Fixo, Newton-Raphson, Secante e Regula Falsi.
# 
# O programa lê os parâmetros de 'entrada.txt' e grava o resultado em 'resultado.txt'.
#
# Formato esperado no arquivo de entrada (entrada.txt):
# metodo a b x0 x1 tol maxIter
#
# Onde:
# - metodo: 1=Bisseção, 2=Ponto Fixo, 3=Newton-Raphson, 4=Secante, 5=Regula Falsi
# - para métodos que não usam alguns campos, mantenha valores (ex.: x1=0)
#
# Exemplo de linha:
# 1 0 1 0 0 1e-6 100
# ===============================================================

import math
import time
import sys

# ===============================================================
# Função do problema (exemplo genérico)
# ===============================================================
def f(x):
    """Função f(x) do problema."""
    return x**3 - 9*x + 3.0

def f_derivada(x):
    """Derivada de f(x)."""
    return 3.0*x*x - 9.0

def phi(x):
    """Função phi(x) usada no método do ponto fixo (exemplo simples)."""
    valor = 9.0*x - 3.0
    # raiz cúbica que também funciona com números negativos
    if valor >= 0:
        return valor ** (1.0/3.0)
    else:
        return - (abs(valor) ** (1.0/3.0))

# ===============================================================
# Estrutura para guardar os dados lidos do arquivo
# ===============================================================
class DadosEntrada:
    def __init__(self, metodo, a, b, x0, x1, tol, max_iter):
        self.metodo = metodo
        self.a = a
        self.b = b
        self.x0 = x0
        self.x1 = x1
        self.tol = tol
        self.max_iter = max_iter


def ler_dados(nome_arquivo="entrada.txt"):
    """Lê os valores do arquivo de entrada e valida as informações."""
    try:
        with open(nome_arquivo, "r", encoding="utf-8") as f:
            conteudo = f.read().strip().split()
            if len(conteudo) < 7:
                raise ValueError("Arquivo precisa conter 7 valores: metodo a b x0 x1 tol maxIter")
            metodo = int(conteudo[0])
            a = float(conteudo[1])
            b = float(conteudo[2])
            x0 = float(conteudo[3])
            x1 = float(conteudo[4])
            tol = float(conteudo[5])
            max_iter = int(conteudo[6])
    except FileNotFoundError:
        print("Erro: arquivo 'entrada.txt' não encontrado.")
        sys.exit(1)
    except Exception as e:
        print("Erro ao ler 'entrada.txt':", e)
        sys.exit(1)

    if metodo < 1 or metodo > 5:
        print("Método inválido! Escolha entre 1 e 5.")
        sys.exit(1)
    if tol <= 0 or max_iter <= 0:
        print("Tolerância e número máximo de iterações devem ser positivos.")
        sys.exit(1)

    return DadosEntrada(metodo, a, b, x0, x1, tol, max_iter)

# ===============================================================
# Funções de formatação de saída
# ===============================================================
def salvar_cabecalho(saida, titulo):
    """Escreve o cabeçalho do método no arquivo de saída."""
    saida.write(f"\n=== {titulo} ===\n")
    saida.write("Iter |       xk        |      f(xk)       |     Erro\n")
    saida.write("-----------------------------------------------------------\n")

def salvar_iteracao(saida, iteracao, x, fx, erro):
    """Escreve uma linha formatada com os dados de cada iteração."""
    saida.write(f"{iteracao:4d} | {x:14.8f} | {fx:14.8f} | {erro:14.8f}\n")

def finalizar_metodo(saida, inicio, atingiu_max_iter, erro, tol):
    """Mensagens finais e cálculo do tempo de execução."""
    if atingiu_max_iter and erro > tol:
        saida.write("\nATENÇÃO: Método atingiu o número máximo de iterações e pode não ter convergido.\n")
        print("⚠️  Atenção: Método atingiu o número máximo de iterações e pode não ter convergido.")
    tempo = time.perf_counter() - inicio
    saida.write(f"\nTempo de execução: {tempo:.6f} segundos\n")
    print(f"Tempo de execução: {tempo:.6f} s")

# ===============================================================
# Métodos Numéricos
# ===============================================================

# 1. Bisseção
def metodo_bissecao(dados, saida):
    salvar_cabecalho(saida, "Método da Bisseção")
    inicio = time.perf_counter()

    a, b = dados.a, dados.b
    if f(a) * f(b) > 0:
        saida.write("Intervalo inválido: f(a)*f(b) > 0\n")
        print("Intervalo inválido: f(a)*f(b) > 0")
        return

    iteracao = 0
    erro = abs(b - a) / 2.0

    while erro > dados.tol and iteracao < dados.max_iter:
        xm = (a + b) / 2.0
        erro = abs(b - a) / 2.0
        iteracao += 1
        salvar_iteracao(saida, iteracao, xm, f(xm), erro)
        if f(a) * f(xm) < 0:
            b = xm
        else:
            a = xm

    finalizar_metodo(saida, inicio, iteracao == dados.max_iter, erro, dados.tol)


# 2. Ponto Fixo
def metodo_ponto_fixo(dados, saida):
    salvar_cabecalho(saida, "Método do Ponto Fixo")
    inicio = time.perf_counter()

    x0 = dados.x0
    iteracao = 0
    erro = float('inf')

    while erro > dados.tol and iteracao < dados.max_iter:
        x1 = phi(x0)
        erro = abs(x1 - x0)
        iteracao += 1
        salvar_iteracao(saida, iteracao, x1, f(x1), erro)
        x0 = x1

    if erro > dados.tol:
        saida.write("\nAviso: Método do Ponto Fixo pode não convergir (|phi'(x)| ≥ 1).\n")
        print("⚠️  Método do Ponto Fixo pode não convergir (|phi'(x)| ≥ 1).")

    finalizar_metodo(saida, inicio, iteracao == dados.max_iter, erro, dados.tol)


# 3. Newton-Raphson
def metodo_newton_raphson(dados, saida):
    salvar_cabecalho(saida, "Método de Newton-Raphson")
    inicio = time.perf_counter()

    x0 = dados.x0
    iteracao = 0
    erro = float('inf')

    while erro > dados.tol and iteracao < dados.max_iter:
        fx = f(x0)
        fdx = f_derivada(x0)
        if abs(fdx) < 1e-12:
            saida.write("Derivada próxima de zero. Encerrando.\n")
            print("Derivada próxima de zero. Encerrando.")
            return
        x1 = x0 - fx / fdx
        erro = abs(x1 - x0)
        iteracao += 1
        salvar_iteracao(saida, iteracao, x1, f(x1), erro)
        x0 = x1

    finalizar_metodo(saida, inicio, iteracao == dados.max_iter, erro, dados.tol)


# 4. Secante
def metodo_secante(dados, saida):
    salvar_cabecalho(saida, "Método da Secante")
    inicio = time.perf_counter()

    x0, x1 = dados.x0, dados.x1
    iteracao = 0
    erro = float('inf')

    while erro > dados.tol and iteracao < dados.max_iter:
        fx0, fx1 = f(x0), f(x1)
        if abs(fx1 - fx0) < 1e-12:
            saida.write("Divisão por zero detectada. Encerrando.\n")
            print("Divisão por zero detectada. Encerrando.")
            return
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        erro = abs(x2 - x1)
        iteracao += 1
        salvar_iteracao(saida, iteracao, x2, f(x2), erro)
        x0, x1 = x1, x2

    finalizar_metodo(saida, inicio, iteracao == dados.max_iter, erro, dados.tol)


# 5. Regula Falsi
def metodo_regula_falsi(dados, saida):
    salvar_cabecalho(saida, "Método da Regula Falsi")
    inicio = time.perf_counter()

    a, b = dados.a, dados.b
    if f(a) * f(b) > 0:
        saida.write("Intervalo inválido: f(a)*f(b) > 0\n")
        print("Intervalo inválido: f(a)*f(b) > 0")
        return

    iteracao = 0
    erro = float('inf')

    while erro > dados.tol and iteracao < dados.max_iter:
        fa, fb = f(a), f(b)
        x = (a * fb - b * fa) / (fb - fa)
        erro = abs(f(x))
        iteracao += 1
        salvar_iteracao(saida, iteracao, x, f(x), erro)
        if fa * f(x) < 0:
            b = x
        else:
            a = x

    finalizar_metodo(saida, inicio, iteracao == dados.max_iter, erro, dados.tol)

# ===============================================================
# Função principal (main)
# ===============================================================
def main():
    dados = ler_dados("entrada.txt")

    try:
        saida = open("resultado.txt", "w", encoding="utf-8")
    except Exception as e:
        print("Erro ao criar arquivo de saída:", e)
        sys.exit(1)

    print("====================================")
    print("     MÉTODOS DE CÁLCULO NUMÉRICO")
    print("====================================")
    print("Método selecionado: ", end="")

    if dados.metodo == 1:
        print("BISSEÇÃO")
        saida.write("Método selecionado: BISSEÇÃO\n")
        metodo_bissecao(dados, saida)
    elif dados.metodo == 2:
        print("PONTO FIXO")
        saida.write("Método selecionado: PONTO FIXO\n")
        metodo_ponto_fixo(dados, saida)
    elif dados.metodo == 3:
        print("NEWTON-RAPHSON")
        saida.write("Método selecionado: NEWTON-RAPHSON\n")
        metodo_newton_raphson(dados, saida)
    elif dados.metodo == 4:
        print("SECANTE")
        saida.write("Método selecionado: SECANTE\n")
        metodo_secante(dados, saida)
    elif dados.metodo == 5:
        print("REGULA FALSI")
        saida.write("Método selecionado: REGULA FALSI\n")
        metodo_regula_falsi(dados, saida)

    print("------------------------------------")
    print("Execução concluída!")
    print("Resultados salvos em resultado.txt")
    print("------------------------------------")
    saida.close()


if __name__ == "__main__":
    main()
