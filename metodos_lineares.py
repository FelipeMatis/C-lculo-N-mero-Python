# Rotinas para resolver sistemas lineares.
# Retornos:
#   x, tempo, status
# ou
#   x, tempo, status, steps  (quando return_steps=True)

import numpy as np
import time

EPS = 1e-18

def is_square(A):
    # verifica se A é matriz quadrada
    A = np.array(A)
    return A.ndim == 2 and A.shape[0] == A.shape[1]

def is_positive_definite(A):
    # tenta fazer Cholesky para checar se é definida positiva
    A = np.array(A, dtype=float)
    if not is_square(A):
        return False
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

# -------------------------
# Função auxiliar para retorno consistente
# -------------------------
def _wrap_return(x, tempo, status, steps=None, return_steps=False):
    # se o chamador quis os passos, devolve também o dicionário steps
    if return_steps:
        return x, tempo, status, steps
    else:
        return x, tempo, status

# -------------------------
# Eliminação de Gauss (sem pivoteamento)
# -------------------------
def eliminacao_gauss(A, b, return_steps=False, show_steps_matrix=False, **kwargs):
    inicio = time.time()
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)
    steps = {"matrizes": [], "actions": []}

    if not is_square(A) or A.shape[0] != b.shape[0]:
        status = "ERRO: A não é quadrada ou dimensões incompatíveis com b."
        return _wrap_return(None, 0.0, status, steps, return_steps)

    n = A.shape[0]
    M = np.hstack([A.copy(), b.reshape(-1,1)])
    if show_steps_matrix:
        steps["matrizes"].append(("Inicial (A|b)", M.copy()))

    # fase de eliminação (sem trocar linhas)
    for i in range(n):
        if abs(M[i, i]) < EPS:
            status = f"ERRO: Pivô (linha {i}) muito próximo de zero — pivoteamento necessário."
            return _wrap_return(None, time.time()-inicio, status, steps, return_steps)
        for j in range(i+1, n):
            m = M[j, i] / M[i, i]
            M[j, i:] = M[j, i:] - m * M[i, i:]
            if show_steps_matrix:
                steps["actions"].append(f"Eliminou linha {j} usando linha {i} (multiplicador={m:.6g})")
                steps["matrizes"].append((f"Depois eliminação i={i}, j={j}", M.copy()))

    # retrosubstituição
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        denom = M[i, i]
        if abs(denom) < EPS:
            status = f"ERRO: Pivô zero durante retrosubstituição na linha {i}."
            return _wrap_return(None, time.time()-inicio, status, steps, return_steps)
        x[i] = (M[i, n] - np.dot(M[i, i+1:n], x[i+1:n])) / denom

    tempo = time.time() - inicio
    status = "Sucesso (Eliminação de Gauss sem pivoteamento)"
    return _wrap_return(x, tempo, status, steps, return_steps)

# -------------------------
# Pivoteamento parcial (troca de linhas)
# -------------------------
def pivoteamento_parcial(A, b, return_steps=False, show_steps_matrix=False, **kwargs):
    inicio = time.time()
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)
    steps = {"matrizes": [], "actions": []}

    if not is_square(A) or A.shape[0] != b.shape[0]:
        status = "ERRO: A não é quadrada ou dimensões incompatíveis com b."
        return _wrap_return(None, 0.0, status, steps, return_steps)

    n = len(b)
    M = np.hstack([A.copy(), b.reshape(-1,1)])
    if show_steps_matrix:
        steps["matrizes"].append(("Inicial (A|b)", M.copy()))

    for i in range(n):
        pivot_row = np.argmax(np.abs(M[i:, i])) + i
        steps["actions"].append(f"Pivô escolhido (linha {pivot_row}) para coluna {i}")
        if abs(M[pivot_row, i]) < EPS:
            status = f"ERRO: Pivô zero (ou quase) na coluna {i}."
            return _wrap_return(None, time.time()-inicio, status, steps, return_steps)
        if pivot_row != i:
            M[[i, pivot_row], :] = M[[pivot_row, i], :]
            steps["actions"].append(f"Trocou linha {i} com {pivot_row}")
            if show_steps_matrix:
                steps["matrizes"].append((f"Após troca linhas {i}<->{pivot_row}", M.copy()))
        for j in range(i+1, n):
            m = M[j, i] / M[i, i]
            M[j, i:] = M[j, i:] - m * M[i, i:]
            steps["actions"].append(f"Eliminou linha {j} usando linha {i} (m={m:.6g})")
            if show_steps_matrix:
                steps["matrizes"].append((f"Depois eliminação i={i}, j={j}", M.copy()))

    # retrosubstituição
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        denom = M[i, i]
        if abs(denom) < EPS:
            status = f"ERRO: Pivô zero na retrosubstituição na linha {i}."
            return _wrap_return(None, time.time()-inicio, status, steps, return_steps)
        x[i] = (M[i, n] - np.dot(M[i, i+1:n], x[i+1:n])) / denom

    tempo = time.time() - inicio
    status = "Sucesso (Gauss com pivoteamento parcial)"
    return _wrap_return(x, tempo, status, steps, return_steps)

# -------------------------
# Pivoteamento completo (linhas e colunas)
# -------------------------
def pivoteamento_completo(A, b, return_steps=False, show_steps_matrix=False, show_permutation=False, **kwargs):
    inicio = time.time()
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)
    steps = {"matrizes": [], "actions": [], "col_perm": None}

    if not is_square(A) or A.shape[0] != b.shape[0]:
        status = "ERRO: Matriz A não é quadrada ou a ordem é inconsistente com b."
        return _wrap_return(None, 0.0, status, steps, return_steps)

    n = len(b)
    M = np.hstack([A.copy(), b.reshape(-1,1)])  # matriz aumentada
    col_perm = list(range(n))  # acompanha permutação de colunas
    if show_steps_matrix:
        steps["matrizes"].append(("Inicial (A|b)", M.copy()))

    for i in range(n):
        sub = np.abs(M[i:, i:n])
        if sub.size == 0:
            status = "ERRO: Submatriz vazia durante pivoteamento completo."
            return _wrap_return(None, time.time()-inicio, status, steps, return_steps)

        local_max_idx = np.unravel_index(np.argmax(sub, axis=None), sub.shape)
        pivot_row = i + local_max_idx[0]
        pivot_col = i + local_max_idx[1]
        steps["actions"].append(f"Pivô absoluto em posição ({pivot_row},{pivot_col}) para etapa {i}")

        if abs(M[pivot_row, pivot_col]) < EPS:
            status = f"ERRO: Pivô zero (ou quase) encontrado na etapa {i+1}. Matriz singular."
            return _wrap_return(None, time.time()-inicio, status, steps, return_steps)

        if pivot_row != i:
            M[[i, pivot_row], :] = M[[pivot_row, i], :]
            steps["actions"].append(f"Trocou linhas {i} <-> {pivot_row}")
            if show_steps_matrix:
                steps["matrizes"].append((f"Após troca linhas {i}<->{pivot_row}", M.copy()))

        if pivot_col != i:
            M[:, [i, pivot_col]] = M[:, [pivot_col, i]]
            col_perm[i], col_perm[pivot_col] = col_perm[pivot_col], col_perm[i]
            steps["actions"].append(f"Trocou colunas {i} <-> {pivot_col} (permutações atualizadas)")
            if show_steps_matrix:
                steps["matrizes"].append((f"Após troca colunas {i}<->{pivot_col}", M.copy()))

        for j in range(i+1, n):
            multiplicador = M[j, i] / M[i, i]
            M[j, i:] = M[j, i:] - multiplicador * M[i, i:]
            steps["actions"].append(f"Eliminou linha {j} usando linha {i} (mult={multiplicador:.6g})")
            if show_steps_matrix:
                steps["matrizes"].append((f"Depois eliminação i={i}, j={j}", M.copy()))

    # retrosubstituição na matriz aumentada (solução permutada)
    x_perm = np.zeros(n)
    for i in range(n-1, -1, -1):
        if abs(M[i, i]) < EPS:
            status = "ERRO: Pivô zero durante retrosubstituição (U com pivô zero)."
            return _wrap_return(None, time.time()-inicio, status, steps, return_steps)
        x_perm[i] = (M[i, n] - np.dot(M[i, i+1:n], x_perm[i+1:n])) / M[i, i]

    # reordena a solução conforme permutação de colunas
    x = np.zeros(n)
    for i_col_after in range(n):
        orig_col_index = col_perm[i_col_after]
        x[orig_col_index] = x_perm[i_col_after]

    tempo = time.time() - inicio
    status = "Sucesso (Gauss com pivoteamento completo)"
    if show_permutation:
        steps["col_perm"] = col_perm.copy()
    return _wrap_return(x, tempo, status, steps, return_steps)

# -------------------------
# Fatoração LU (sem pivoteamento)
# -------------------------
def fatoracao_lu(A, b, return_steps=False, show_steps_matrix=False, show_LU=False, **kwargs):
    inicio = time.time()
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)
    steps = {"matrizes": [], "actions": [], "L": None, "U": None}

    if not is_square(A) or A.shape[0] != b.shape[0]:
        status = "ERRO: A não é quadrada ou dimensões incompatíveis com b."
        return _wrap_return(None, 0.0, status, steps, return_steps)

    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    if show_steps_matrix:
        steps["matrizes"].append(("Inicial U", U.copy()))
        steps["matrizes"].append(("Inicial L", L.copy()))

    for k in range(n):
        if abs(U[k, k]) < EPS:
            status = f"ERRO: Pivô zero em U[{k},{k}]. LU sem pivoteamento não aplicável."
            return _wrap_return(None, time.time()-inicio, status, steps, return_steps)
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] = U[i, k:] - L[i, k] * U[k, k:]
            steps["actions"].append(f"Calculou L[{i},{k}] = {L[i,k]:.6g} e atualizou U linha {i}")
            if show_steps_matrix:
                steps["matrizes"].append((f"Após k={k}, atualização U", U.copy()))
                steps["matrizes"].append((f"Após k={k}, L", L.copy()))

    # resolve Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    # resolve Ux = y
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        if abs(U[i, i]) < EPS:
            status = f"ERRO: Pivô zero em U na linha {i}."
            return _wrap_return(None, time.time()-inicio, status, steps, return_steps)
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

    tempo = time.time() - inicio
    status = "Sucesso (Fatoração LU sem pivoteamento)."
    if show_LU:
        steps["L"] = L.copy()
        steps["U"] = U.copy()
    return _wrap_return(x, tempo, status, steps, return_steps)

# -------------------------
# Cholesky
# -------------------------
def cholesky(A, b, return_steps=False, show_steps_matrix=False, show_L=False, **kwargs):
    inicio = time.time()
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)
    steps = {"matrizes": [], "actions": [], "L": None}

    if not is_square(A) or A.shape[0] != b.shape[0]:
        status = "ERRO: A não é quadrada ou dimensões incompatíveis com b."
        return _wrap_return(None, 0.0, status, steps, return_steps)

    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        status = "ERRO: Cholesky não aplicável — matriz não é definida positiva."
        return _wrap_return(None, time.time()-inicio, status, steps, return_steps)

    # resolve Ly = b
    y = np.zeros_like(b, dtype=float)
    n = len(b)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    # resolve L^T x = y
    x = np.zeros_like(b, dtype=float)
    LT = L.T
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(LT[i, i+1:], x[i+1:])) / LT[i, i]

    tempo = time.time() - inicio
    status = "Sucesso (Cholesky)."
    if show_L:
        steps["L"] = L.copy()
    return _wrap_return(x, tempo, status, steps, return_steps)

# -------------------------
# Gauss-Jacobi (iterativo)
# -------------------------
def gauss_jacobi(A, b, x0=None, tol=1e-8, max_iter=100, return_steps=False, record_iterations=False, **kwargs):
    inicio = time.time()
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)
    steps = {"iterations": [], "actions": []}
    n = b.shape[0]
    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.array(x0, dtype=float).reshape(-1)
        if x.shape[0] != n:
            x = np.zeros(n)

    if not is_square(A) or A.shape[0] != n:
        status = "ERRO: A não é quadrada ou dimensões inconsistentes."
        return _wrap_return(None, 0.0, status, steps, return_steps)

    D = np.diag(A)
    if np.any(np.abs(D) < EPS):
        status = "ERRO: Zero na diagonal torna Jacobi impraticável (divisão por zero)."
        return _wrap_return(None, 0.0, status, steps, return_steps)

    R = A - np.diagflat(D)
    x_new = x.copy()

    for k in range(1, max_iter+1):
        x_new = (b - np.dot(R, x)) / D
        if record_iterations:
            steps["iterations"].append(x_new.copy())
            steps["actions"].append(f"Iteração {k}")
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            tempo = time.time() - inicio
            status = f"Convergiu em {k} iterações (Gauss-Jacobi)."
            return _wrap_return(x_new, tempo, status, steps, return_steps)
        x = x_new.copy()

    tempo = time.time() - inicio
    status = "Atenção: Não convergiu dentro do número máximo de iterações (Jacobi)."
    return _wrap_return(x_new, tempo, status, steps, return_steps)

# -------------------------
# Gauss-Seidel (iterativo)
# -------------------------
def gauss_seidel(A, b, x0=None, tol=1e-8, max_iter=100, return_steps=False, record_iterations=False, **kwargs):
    inicio = time.time()
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)
    steps = {"iterations": [], "actions": []}
    n = b.shape[0]
    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.array(x0, dtype=float).reshape(-1)
        if x.shape[0] != n:
            x = np.zeros(n)

    if not is_square(A) or A.shape[0] != n:
        status = "ERRO: A não é quadrada ou dimensões inconsistentes."
        return _wrap_return(None, 0.0, status, steps, return_steps)

    for k in range(1, max_iter+1):
        x_old = x.copy()
        for i in range(n):
            soma1 = np.dot(A[i, :i], x[:i])
            soma2 = np.dot(A[i, i+1:], x_old[i+1:])
            denom = A[i, i]
            if abs(denom) < EPS:
                status = f"ERRO: Zero na diagonal em linha {i} (não é possível dividir)."
                return _wrap_return(None, time.time() - inicio, status, steps, return_steps)
            x[i] = (b[i] - soma1 - soma2) / denom

        if record_iterations:
            steps["iterations"].append(x.copy())
            steps["actions"].append(f"Iteração {k}")
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            tempo = time.time() - inicio
            status = f"Convergiu em {k} iterações (Gauss-Seidel)."
            return _wrap_return(x, tempo, status, steps, return_steps)

    tempo = time.time() - inicio
    status = "Atenção: Não convergiu dentro do número máximo de iterações (Gauss-Seidel)."
    return _wrap_return(x, tempo, status, steps, return_steps)

# -------------------------
# Mapeamento usado pela GUI
# -------------------------
METODOS = {
    "Gauss sem pivoteamento": eliminacao_gauss,
    "Gauss com pivoteamento parcial": pivoteamento_parcial,
    "Gauss com pivoteamento completo": pivoteamento_completo,
    "Fatoração LU": fatoracao_lu,
    "Fatoração de Cholesky": cholesky,
    "Método iterativo - Gauss-Jacobi": gauss_jacobi,
    "Método iterativo - Gauss-Seidel": gauss_seidel,
}
