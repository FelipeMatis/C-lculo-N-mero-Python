# metodos_raizes.py
# Tradução do seu código C para Python: Bisseção, Ponto Fixo, Newton-Raphson,
# Secante e Regula Falsi. Lê 'entrada.txt' e escreve 'resultado.txt'.
#
# Formato de entrada esperado (entrada.txt):
# metodo a b x0 x1 tol maxIter
# onde:
# - metodo: 1=bissecao 2=ponto fixo 3=newton 4=secante 5=regula falsi
# - para métodos que não usam alguns campos, mantenha valores (ex.: x1=0)
#
# Exemplo de linha:
# 1 0 1 0 0 1e-6 100

import math
import time
import sys

# =====================
# Função do problema (edite se quiser)
# =====================
def f(x):
    return x**3 - 9*x + 3.0

def f_derivada(x):
    return 3.0*x*x - 9.0

def phi(x):
    # exemplo de iteração por ponto fixo compatível com phi(x) = cbrt(9x - 3)
    # atenção: nem sempre converge; é só um exemplo transliterado do C
    val = 9.0*x - 3.0
    # cbrt para negativos também
    if val >= 0:
        return val ** (1.0/3.0)
    else:
        return - (abs(val) ** (1.0/3.0))

# =====================
# Estrutura de leitura
# =====================
class DadosEntrada:
    def __init__(self, metodo, a, b, x0, x1, tol, maxIter):
        self.metodo = metodo
        self.a = a
        self.b = b
        self.x0 = x0
        self.x1 = x1
        self.tol = tol
        self.maxIter = maxIter

def lerDados(nomeArquivo="entrada.txt"):
    try:
        with open(nomeArquivo, "r", encoding="utf-8") as f:
            # ler primeiro número por validade
            conteudo = f.read().strip().split()
            if len(conteudo) < 7:
                raise ValueError("Arquivo de entrada precisa conter 7 valores: metodo a b x0 x1 tol maxIter")
            arr = conteudo
            metodo = int(arr[0])
            a = float(arr[1])
            b = float(arr[2])
            x0 = float(arr[3])
            x1 = float(arr[4])
            tol = float(arr[5])
            maxIter = int(arr[6])
    except FileNotFoundError:
        print("Erro: arquivo 'entrada.txt' nao encontrado.")
        sys.exit(1)
    except Exception as e:
        print("Erro ao ler entrada.txt:", e)
        sys.exit(1)

    if metodo < 1 or metodo > 5:
        print("Metodo invalido! Escolha entre 1 e 5.")
        sys.exit(1)
    if tol <= 0 or maxIter <= 0:
        print("Tolerancia e numero maximo de iteracoes devem ser positivos.")
        sys.exit(1)

    return DadosEntrada(metodo, a, b, x0, x1, tol, maxIter)

# =====================
# Auxiliares de saída
# =====================
def salvar_cabecalho(saida, titulo):
    saida.write("\n=== {} ===\n".format(titulo))
    saida.write("Iter |       xk        |      f(xk)       |     Erro\n")
    saida.write("-----------------------------------------------------------\n")

def salvar_iteracao(saida, iter, x, fx, erro):
    saida.write("{:4d} | {:14.8f} | {:14.8f} | {:14.8f}\n".format(iter, x, fx, erro))

def finalizar_metodo(saida, inicio, maxIterAlcancado, erro, tol):
    if maxIterAlcancado and erro > tol:
        saida.write("\nATENCAO: Metodo atingiu o numero maximo de iteracoes e pode nao ter convergido.\n")
        print("⚠️  Aviso: Metodo atingiu o numero maximo de iteracoes e pode nao ter convergido.")
    tempo = time.perf_counter() - inicio
    saida.write("\nTempo de execucao: {:.6f} segundos\n".format(tempo))
    print("Tempo de execucao: {:.6f} s".format(tempo))

# =====================
# Métodos
# =====================

# 1. Bisseção
def bissecao(d, saida):
    salvar_cabecalho(saida, "Metodo da Bissecao")
    inicio = time.perf_counter()

    a = d.a; b = d.b
    if f(a) * f(b) > 0:
        saida.write("Intervalo invalido: f(a)*f(b) > 0\n")
        print("Intervalo invalido: f(a)*f(b) > 0")
        return
    i = 0
    erro = abs(b - a) / 2.0
    while erro > d.tol and i < d.maxIter:
        xm = (a + b) / 2.0
        erro = abs(b - a) / 2.0
        i += 1
        salvar_iteracao(saida, i, xm, f(xm), erro)
        if f(a) * f(xm) < 0:
            b = xm
        else:
            a = xm
    finalizar_metodo(saida, inicio, i == d.maxIter, erro, d.tol)

# 2. Ponto Fixo
def ponto_fixo(d, saida):
    salvar_cabecalho(saida, "Metodo do Ponto Fixo")
    inicio = time.perf_counter()
    x0 = d.x0
    i = 0
    erro = float('inf')
    while erro > d.tol and i < d.maxIter:
        x1 = phi(x0)
        erro = abs(x1 - x0)
        i += 1
        salvar_iteracao(saida, i, x1, f(x1), erro)
        x0 = x1
    if erro > d.tol:
        saida.write("\nAviso: Metodo do Ponto Fixo pode nao convergir (|phi'(x)| >= 1).\n")
        print("⚠️  Metodo do Ponto Fixo pode nao convergir (|phi'(x)| >= 1).")
    finalizar_metodo(saida, inicio, i == d.maxIter, erro, d.tol)

# 3. Newton-Raphson
def newton(d, saida):
    salvar_cabecalho(saida, "Metodo de Newton-Raphson")
    inicio = time.perf_counter()
    x0 = d.x0
    i = 0
    erro = float('inf')
    while erro > d.tol and i < d.maxIter:
        fx = f(x0)
        fdx = f_derivada(x0)
        if abs(fdx) < 1e-12:
            saida.write("Derivada proxima de zero. Encerrando.\n")
            print("Derivada proxima de zero. Encerrando.")
            return
        x1 = x0 - fx / fdx
        erro = abs(x1 - x0)
        i += 1
        salvar_iteracao(saida, i, x1, f(x1), erro)
        x0 = x1
    finalizar_metodo(saida, inicio, i == d.maxIter, erro, d.tol)

# 4. Secante
def secante(d, saida):
    salvar_cabecalho(saida, "Metodo da Secante")
    inicio = time.perf_counter()
    x0 = d.x0; x1 = d.x1
    i = 0
    erro = float('inf')
    while erro > d.tol and i < d.maxIter:
        fx0 = f(x0); fx1 = f(x1)
        if abs(fx1 - fx0) < 1e-12:
            saida.write("Divisao por zero detectada. Encerrando.\n")
            print("Divisao por zero detectada. Encerrando.")
            return
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        erro = abs(x2 - x1)
        i += 1
        salvar_iteracao(saida, i, x2, f(x2), erro)
        x0, x1 = x1, x2
    finalizar_metodo(saida, inicio, i == d.maxIter, erro, d.tol)

# 5. Regula Falsi
def regula_falsi(d, saida):
    salvar_cabecalho(saida, "Metodo da Regula Falsi")
    inicio = time.perf_counter()
    a = d.a; b = d.b
    if f(a) * f(b) > 0:
        saida.write("Intervalo invalido: f(a)*f(b) > 0\n")
        print("Intervalo invalido: f(a)*f(b) > 0")
        return
    i = 0
    erro = float('inf')
    while erro > d.tol and i < d.maxIter:
        fa = f(a); fb = f(b)
        x = (a * fb - b * fa) / (fb - fa)
        erro = abs(f(x))
        i += 1
        salvar_iteracao(saida, i, x, f(x), erro)
        if fa * f(x) < 0:
            b = x
        else:
            a = x
    finalizar_metodo(saida, inicio, i == d.maxIter, erro, d.tol)

# =====================
# Runner (igual ao main do C)
# =====================
def main():
    dados = lerDados("entrada.txt")
    try:
        saida = open("resultado.txt", "w", encoding="utf-8")
    except Exception as e:
        print("Erro ao criar arquivo de saida:", e)
        sys.exit(1)

    print("====================================")
    print("     METODOS DE CALCULO NUMERICO")
    print("====================================")
    print("Metodo selecionado: ", end="")

    if dados.metodo == 1:
        print("BISSECAO")
        saida.write("Metodo selecionado: BISSECAO\n")
        bissecao(dados, saida)
    elif dados.metodo == 2:
        print("PONTO FIXO")
        saida.write("Metodo selecionado: PONTO FIXO\n")
        ponto_fixo(dados, saida)
    elif dados.metodo == 3:
        print("NEWTON-RAPHSON")
        saida.write("Metodo selecionado: NEWTON-RAPHSON\n")
        newton(dados, saida)
    elif dados.metodo == 4:
        print("SECANTE")
        saida.write("Metodo selecionado: SECANTE\n")
        secante(dados, saida)
    elif dados.metodo == 5:
        print("REGULA FALSI")
        saida.write("Metodo selecionado: REGULA FALSI\n")
        regula_falsi(dados, saida)

    print("------------------------------------")
    print("Execucao concluida!")
    print("Resultados salvos em resultado.txt")
    print("------------------------------------")
    saida.close()

if __name__ == "__main__":
    main()
