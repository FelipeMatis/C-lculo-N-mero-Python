# GUI em Tkinter para usar métodos lineares e métodos para encontrar raízes

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import os
import io

from metodos_lineares import METODOS as METODOS_LIN
import metodos_raizes as MR  # funções: bissecao, newton, secante, etc.


class App:
    def __init__(self, root):
        # janela principal e variáveis principais
        self.root = root
        self.root.title("Trabalho - Métodos do Capítulo 3")
        self.A = None  # matriz do sistema
        self.b = None  # vetor do sistema

        self._build_ui()

    def _build_ui(self):
        # frame principal (agora como atributo para facilitar configuração)
        self.frm = ttk.Frame(self.root, padding=8)
        self.frm.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # === Parte: carregar A e b ===
        load_frame = ttk.LabelFrame(self.frm, text="Carregar Sistema (A e b)")
        load_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        load_frame.columnconfigure(0, weight=1)
        load_frame.columnconfigure(1, weight=1)
        load_frame.columnconfigure(2, weight=1)

        ttk.Button(load_frame, text="Carregar arquivo (A|b)", command=self.load_ab_file).grid(
            row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Button(load_frame, text="Carregar A (apenas coeficientes)",
                   command=self.load_a_file).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(load_frame, text="Carregar b (vetor)", command=self.load_b_file).grid(
            row=0, column=2, padx=5, pady=5, sticky="w")

        self.lbl_status = ttk.Label(load_frame, text="Nenhum arquivo carregado.")
        self.lbl_status.grid(row=1, column=0, columnspan=3, sticky="w", padx=5)

        # caixas de texto para colar A e b direto
        ttk.Label(load_frame, text="Colar A (linha por linha):").grid(
            row=2, column=0, sticky="w", padx=2, pady=(8, 0))
        self.text_a = tk.Text(load_frame, height=5, width=50)
        self.text_a.grid(row=3, column=0, columnspan=2, padx=2, pady=2, sticky="nsew")

        ttk.Label(load_frame, text="Colar b (linha ou coluna):").grid(
            row=2, column=2, sticky="w", padx=2, pady=(8, 0))
        self.text_b = tk.Text(load_frame, height=5, width=20)
        self.text_b.grid(row=3, column=2, padx=2, pady=2, sticky="nsew")

        ttk.Button(load_frame, text="Carregar do texto", command=self.load_from_text).grid(
            row=4, column=0, padx=5, pady=5, sticky="w")

        # === Parte: métodos lineares ===
        method_frame = ttk.LabelFrame(self.frm, text="Métodos Lineares")
        method_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        method_frame.columnconfigure(0, weight=1)
        method_frame.columnconfigure(1, weight=1)

        # combobox com os métodos disponíveis (vem de METODOS_LIN)
        self.metodo_selecionado = tk.StringVar(value=list(METODOS_LIN.keys())[0])
        metodo_combo = ttk.Combobox(method_frame, textvariable=self.metodo_selecionado,
                                    values=list(METODOS_LIN.keys()), state="readonly")
        metodo_combo.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        metodo_combo.bind("<<ComboboxSelected>>", lambda e: self._on_metodo_change())

        # opções específicas do método
        self.options_frame = ttk.Frame(method_frame)
        self.options_frame.grid(row=1, column=0, sticky="ew", pady=2)
        self._build_linear_options()

        # parâmetros genéricos para métodos iterativos
        iter_frame = ttk.LabelFrame(self.frm, text="Parâmetros (métodos iterativos)")
        iter_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        iter_frame.columnconfigure(0, weight=0)
        iter_frame.columnconfigure(1, weight=1)
        iter_frame.columnconfigure(2, weight=0)
        iter_frame.columnconfigure(3, weight=1)

        ttk.Label(iter_frame, text="Tolerância:").grid(row=0, column=0, sticky="w", padx=2)
        self.tol = tk.Entry(iter_frame, width=12)
        self.tol.insert(0, "1e-8")
        self.tol.grid(row=0, column=1, padx=2, sticky="w")

        ttk.Label(iter_frame, text="Máx. iterações:").grid(row=0, column=2, sticky="w", padx=2)
        self.max_iter = tk.Entry(iter_frame, width=12)
        self.max_iter.insert(0, "1000")
        self.max_iter.grid(row=0, column=3, padx=2, sticky="w")

        ttk.Label(iter_frame, text="Chute inicial x0 (separado por vírgula):").grid(
            row=1, column=0, columnspan=2, sticky="w", padx=2)
        self.x0_init = tk.Entry(iter_frame)
        self.x0_init.grid(row=1, column=2, columnspan=2, sticky="ew", padx=2)

        # === Parte: métodos para raízes ===
        roots_frame = ttk.LabelFrame(self.frm, text="Métodos para Raízes")
        roots_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        roots_frame.columnconfigure(0, weight=0)
        roots_frame.columnconfigure(1, weight=1)
        roots_frame.columnconfigure(2, weight=0)
        roots_frame.columnconfigure(3, weight=1)

        ttk.Label(roots_frame, text="Método:").grid(row=0, column=0, sticky="w")
        self.root_method_var = tk.StringVar(value="Bisseção")
        root_methods = ["Bisseção", "Ponto Fixo", "Newton-Raphson", "Secante", "Regula Falsi"]
        root_combo = ttk.Combobox(roots_frame, textvariable=self.root_method_var, values=root_methods, state="readonly")
        root_combo.grid(row=0, column=1, sticky="ew", padx=4)

        # entradas numéricas usadas pelos métodos de raízes
        ttk.Label(roots_frame, text="a:").grid(row=1, column=0, sticky="e")
        self.root_a = tk.Entry(roots_frame, width=12)
        self.root_a.grid(row=1, column=1, sticky="w", padx=4)
        ttk.Label(roots_frame, text="b:").grid(row=1, column=2, sticky="e")
        self.root_b = tk.Entry(roots_frame, width=12)
        self.root_b.grid(row=1, column=3, sticky="w", padx=4)

        ttk.Label(roots_frame, text="x0:").grid(row=2, column=0, sticky="e")
        self.root_x0 = tk.Entry(roots_frame, width=12)
        self.root_x0.grid(row=2, column=1, sticky="w", padx=4)
        ttk.Label(roots_frame, text="x1:").grid(row=2, column=2, sticky="e")
        self.root_x1 = tk.Entry(roots_frame, width=12)
        self.root_x1.grid(row=2, column=3, sticky="w", padx=4)

        ttk.Label(roots_frame, text="Tolerância:").grid(row=3, column=0, sticky="e")
        self.root_tol = tk.Entry(roots_frame, width=12)
        self.root_tol.insert(0, "1e-6")
        self.root_tol.grid(row=3, column=1, sticky="w", padx=4)
        ttk.Label(roots_frame, text="Max iterações:").grid(row=3, column=2, sticky="e")
        self.root_maxiter = tk.Entry(roots_frame, width=12)
        self.root_maxiter.insert(0, "100")
        self.root_maxiter.grid(row=3, column=3, sticky="w", padx=4)

        # opção para ver passos dos métodos de raízes
        self.var_show_roots_steps = tk.BooleanVar(value=False)
        ttk.Checkbutton(roots_frame, text="Mostrar passos", variable=self.var_show_roots_steps).grid(
            row=4, column=0, columnspan=2, sticky="w", padx=4)

        # botões de carregar entrada opcional e executar método de raízes
        ttk.Button(roots_frame, text="Carregar entrada.txt (opcional)", command=self.load_entrada_file).grid(
            row=5, column=0, padx=4, pady=4, sticky="w")
        ttk.Button(roots_frame, text="Resolver Raiz", command=self.run_roots).grid(
            row=5, column=1, padx=4, pady=4, sticky="w")

        # === Botões principais ===
        btn_frame = ttk.Frame(self.frm)
        btn_frame.grid(row=4, column=0, sticky="ew", pady=5)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        btn_frame.columnconfigure(2, weight=1)
        ttk.Button(btn_frame, text="Resolver Método Linear", command=self.resolver_linear).grid(
            row=0, column=0, padx=5, sticky="w")
        ttk.Button(btn_frame, text="Limpar", command=self.limpar).grid(
            row=0, column=1, padx=5, sticky="w")
        ttk.Button(btn_frame, text="Salvar Resultado (.txt)", command=self.salvar_resultado).grid(
            row=0, column=2, padx=5, sticky="w")

        # === Área de resultados (abas) ===
        self.result_nb = ttk.Notebook(self.frm)
        self.result_nb.grid(row=5, column=0, sticky="nsew", padx=5, pady=5)
        self.frm.rowconfigure(5, weight=1)

        # aba: resultados resumidos
        result_frame = ttk.Frame(self.result_nb)
        self.result_nb.add(result_frame, text="Resultados")
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        self.texto_resultado = tk.Text(result_frame, height=20, wrap=tk.NONE)
        self.texto_resultado.grid(row=0, column=0, sticky="nsew")
        vsb = ttk.Scrollbar(result_frame, orient="vertical", command=self.texto_resultado.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        self.texto_resultado.configure(yscrollcommand=vsb.set)
        hsb = ttk.Scrollbar(result_frame, orient="horizontal", command=self.texto_resultado.xview)
        hsb.grid(row=1, column=0, sticky="ew")
        self.texto_resultado.configure(xscrollcommand=hsb.set)

        # aba: passos detalhados
        steps_frame = ttk.Frame(self.result_nb)
        self.result_nb.add(steps_frame, text="Passos / Saída detalhada")
        steps_frame.columnconfigure(0, weight=1)
        steps_frame.rowconfigure(0, weight=1)
        self.texto_passos = tk.Text(steps_frame, height=20, wrap=tk.NONE)
        self.texto_passos.grid(row=0, column=0, sticky="nsew")
        vsb2 = ttk.Scrollbar(steps_frame, orient="vertical", command=self.texto_passos.yview)
        vsb2.grid(row=0, column=1, sticky="ns")
        self.texto_passos.configure(yscrollcommand=vsb2.set)
        hsb2 = ttk.Scrollbar(steps_frame, orient="horizontal", command=self.texto_passos.xview)
        hsb2.grid(row=1, column=0, sticky="ew")
        self.texto_passos.configure(xscrollcommand=hsb2.set)

        self._configure_grid_weights()

    def _configure_grid_weights(self):
        # garante que o frame principal expanda
        try:
            self.root.update_idletasks()
            self.frm.columnconfigure(0, weight=1)
            # assegura que o notebook e sua aba cresçam
            try:
                self.result_nb.grid_configure(sticky="nsew")
                self.result_nb.columnconfigure(0, weight=1)
                self.result_nb.rowconfigure(0, weight=1)
            except Exception:
                pass

            # percorre filhos e define peso mínimo nas colunas/linhas
            def _scan(w):
                try:
                    w.columnconfigure(0, weight=1)
                except Exception:
                    pass
                try:
                    w.rowconfigure(0, weight=1)
                except Exception:
                    pass
                for c in w.winfo_children():
                    _scan(c)

            _scan(self.frm)

            # garantir que Text widgets estejam sticky nsew
            for child in self.frm.winfo_children():
                for sub in child.winfo_children():
                    if isinstance(sub, tk.Text):
                        try:
                            sub.grid_configure(sticky="nsew")
                            parent = sub.master
                            info = sub.grid_info()
                            parent.columnconfigure(info.get('column', 0), weight=1)
                            parent.rowconfigure(info.get('row', 0), weight=1)
                        except Exception:
                            pass
        except Exception:
            pass

    # -------------------------
    # Opções específicas para métodos lineares
    # -------------------------
    def _build_linear_options(self):
        # limpa o que já tem e cria opções úteis
        for w in self.options_frame.winfo_children():
            w.destroy()
        ttk.Label(self.options_frame, text="Opções específicas do método:").grid(
            row=0, column=0, sticky="w")
        self.var_show_steps = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.options_frame, text="Mostrar passos detalhados",
                        variable=self.var_show_steps).grid(row=1, column=0, sticky="w", padx=4)
        self.var_show_matrices = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.options_frame, text="Incluir matrizes em cada passo",
                        variable=self.var_show_matrices).grid(row=2, column=0, sticky="w", padx=4)
        self.var_show_LU = tk.BooleanVar(value=False)
        self.cb_LU = ttk.Checkbutton(
            self.options_frame, text="Exibir L e U (quando aplicável)", variable=self.var_show_LU)
        self.var_show_permutation = tk.BooleanVar(value=False)
        self.cb_perm = ttk.Checkbutton(
            self.options_frame, text="Exibir permutações (pivoteamento)", variable=self.var_show_permutation)
        # ajusta visibilidade conforme o método selecionado
        self._on_metodo_change()

    def _on_metodo_change(self):
        # mostra/oculta opções extras dependendo do nome do método
        metodo = self.metodo_selecionado.get().lower()
        try:
            self.cb_LU.grid_forget()
            self.cb_perm.grid_forget()
        except Exception:
            pass
        if "lu" in metodo or "fatoração lu" in metodo or "fatoracao lu" in metodo:
            self.cb_LU.grid(row=3, column=0, sticky="w", padx=4)
        if "completo" in metodo or "pivoteamento completo" in metodo:
            self.cb_perm.grid(row=3, column=0, sticky="w", padx=4)

    # -------------------------
    # Carregamento de arquivos / texto
    # -------------------------
    def parse_text_matrix(self, txt):
        # transforma texto com números em uma matriz numpy
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip() != ""]
        mat = []
        for ln in lines:
            parts = ln.split()
            mat.append([float(x) for x in parts])
        return np.array(mat, dtype=float)

    def load_ab_file(self):
        # abre arquivo onde a última coluna é b (A|b)
        path = filedialog.askopenfilename(title="Selecionar arquivo A|b", filetypes=[
            ("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            mat = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line == "":
                        continue
                    parts = line.split()
                    mat.append([float(x) for x in parts])
            mat = np.array(mat, dtype=float)
            if mat.ndim != 2 or mat.shape[1] < 2:
                messagebox.showerror(
                    "Erro", "Formato inválido para arquivo estendido (A|b).")
                return
            self.A = mat[:, :-1]
            self.b = mat[:, -1].reshape(-1)
            self.lbl_status.config(
                text=f"Carregado A|b de: {os.path.basename(path)} (A: {self.A.shape}, b: {self.b.shape})")
            self.texto_resultado.insert(
                tk.END, f"Arquivo '{os.path.basename(path)}' carregado como A|b.\n")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao carregar arquivo: {e}")

    def load_a_file(self):
        # abre arquivo com apenas a matriz A (deve ser quadrada)
        path = filedialog.askopenfilename(title="Selecionar arquivo A", filetypes=[
            ("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            mat = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line == "":
                        continue
                    parts = line.split()
                    mat.append([float(x) for x in parts])
            mat = np.array(mat, dtype=float)
            if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
                messagebox.showerror(
                    "Erro", "Formato inválido para matriz A. Deve ser quadrada.")
                return
            self.A = mat
            self.lbl_status.config(
                text=f"Carregado A de: {os.path.basename(path)} (A: {self.A.shape})")
            self.texto_resultado.insert(
                tk.END, f"Arquivo '{os.path.basename(path)}' carregado como A.\n")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao carregar arquivo A: {e}")

    def load_b_file(self):
        # abre arquivo para vetor b (pode ser vários números em linha ou coluna)
        path = filedialog.askopenfilename(title="Selecionar arquivo b", filetypes=[
            ("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            vals = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line == "":
                        continue
                    parts = line.split()
                    for x in parts:
                        vals.append(float(x))
            self.b = np.array(vals, dtype=float).reshape(-1)
            self.lbl_status.config(
                text=f"Carregado b de: {os.path.basename(path)} (b: {self.b.shape})")
            self.texto_resultado.insert(
                tk.END, f"Arquivo '{os.path.basename(path)}' carregado como b.\n")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao carregar arquivo b: {e}")

    def load_from_text(self):
        # pega A e/ou b a partir das caixas de texto da interface
        a_txt = self.text_a.get("1.0", tk.END).strip()
        b_txt = self.text_b.get("1.0", tk.END).strip()
        try:
            if a_txt == "" and b_txt == "":
                messagebox.showinfo("Info", "Cole A ou b antes de carregar.")
                return

            if a_txt != "":
                A = self.parse_text_matrix(a_txt)
                if A.ndim != 2:
                    messagebox.showerror("Erro", "Formato inválido para A no texto.")
                    return

                # se tem coluna a mais e não passou b, trata como A|b
                if b_txt == "" and A.shape[1] == A.shape[0] + 1:
                    self.A = A[:, :-1]
                    self.b = A[:, -1].reshape(-1)
                else:
                    # se só A foi colado, exige quadrada
                    if A.shape[0] != A.shape[1]:
                        messagebox.showerror(
                            "Erro", "Se estiver carregando apenas A pelo texto, A deve ser quadrada (n x n).")
                        return
                    self.A = A

            if b_txt != "":
                parts = []
                for ln in b_txt.splitlines():
                    for x in ln.split():
                        parts.append(float(x))
                self.b = np.array(parts, dtype=float).reshape(-1)

            self.lbl_status.config(
                text=f"Carregado A e/ou b do texto (A: {self.A.shape if self.A is not None else None}, b: {self.b.shape if self.b is not None else None})")
            self.texto_resultado.insert(tk.END, "Dados carregados do texto.\n")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao carregar do texto: {e}")

    # -------------------------
    # Carregar arquivo de entrada para métodos de raízes (opcional)
    # -------------------------
    def load_entrada_file(self):
        # espera arquivo com 7 valores: metodo a b x0 x1 tol maxIter
        path = filedialog.askopenfilename(title="Selecionar entrada.txt", filetypes=[
            ("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                tokens = f.read().strip().split()
                if len(tokens) < 7:
                    messagebox.showerror(
                        "Erro", "entrada.txt precisa ter 7 valores: metodo a b x0 x1 tol maxIter")
                    return
                metodo = int(tokens[0])
                a = tokens[1]
                b = tokens[2]
                x0 = tokens[3]
                x1 = tokens[4]
                tol = tokens[5]
                maxit = tokens[6]

                self.root_method_var.set(
                    ["Bisseção", "Ponto Fixo", "Newton-Raphson", "Secante", "Regula Falsi"][metodo-1])
                self.root_a.delete(0, tk.END)
                self.root_a.insert(0, a)
                self.root_b.delete(0, tk.END)
                self.root_b.insert(0, b)
                self.root_x0.delete(0, tk.END)
                self.root_x0.insert(0, x0)
                self.root_x1.delete(0, tk.END)
                self.root_x1.insert(0, x1)
                self.root_tol.delete(0, tk.END)
                self.root_tol.insert(0, tol)
                self.root_maxiter.delete(0, tk.END)
                self.root_maxiter.insert(0, maxit)
                self.texto_resultado.insert(tk.END, f"entrada.txt carregado: {os.path.basename(path)}\n")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao carregar entrada.txt: {e}")

    # -------------------------
    # Executa métodos lineares (usa METODOS_LIN)
    # -------------------------
    def resolver_linear(self):
        # limpa áreas de saída
        self.texto_resultado.delete('1.0', tk.END)
        self.texto_passos.delete('1.0', tk.END)

        if self.A is None or self.b is None:
            messagebox.showerror("Erro", "Por favor, carregue a matriz A e o vetor b.")
            return

        metodo_nome = self.metodo_selecionado.get()
        metodo_func = METODOS_LIN[metodo_nome]

        show_steps = self.var_show_steps.get() if hasattr(self, "var_show_steps") else False
        show_matrices = self.var_show_matrices.get() if hasattr(self, "var_show_matrices") else False
        show_LU = self.var_show_LU.get() if hasattr(self, "var_show_LU") else False
        show_perm = self.var_show_permutation.get() if hasattr(self, "var_show_permutation") else False

        try:
            # se o método for iterativo, pega a tolerância, max iterações e chute inicial
            if "iterativo" in metodo_nome.lower():
                try:
                    tol = float(self.tol.get())
                    max_iter = int(self.max_iter.get())
                except Exception:
                    messagebox.showerror("Erro de Entrada", "Tolerância ou número máximo inválido.")
                    return

                x0_str = self.x0_init.get().strip()
                if x0_str == "":
                    x0 = np.zeros(self.b.shape[0])
                else:
                    try:
                        x0_parts = [float(x.strip()) for x in x0_str.split(',') if x.strip() != ""]
                        if len(x0_parts) != self.b.shape[0]:
                            x0 = np.zeros(self.b.shape[0])
                            L = min(len(x0_parts), len(x0))
                            x0[:L] = x0_parts[:L]
                        else:
                            x0 = np.array(x0_parts, dtype=float)
                    except Exception:
                        x0 = np.zeros(self.b.shape[0])

                sol = metodo_func(self.A, self.b, x0=x0, tol=tol, max_iter=max_iter,
                                  return_steps=show_steps, record_iterations=show_steps)
            else:
                # método direto
                sol = metodo_func(self.A, self.b,
                                  return_steps=show_steps,
                                  show_steps_matrix=show_matrices,
                                  show_LU=show_LU,
                                  show_permutation=show_perm)
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao executar método linear: {e}")
            return

        # interpreta retorno (com ou sem passos)
        if show_steps:
            try:
                x, tempo, status, steps = sol
            except Exception:
                messagebox.showerror("Erro", "Formato retornado inesperado.")
                return
        else:
            try:
                x, tempo, status = sol
                steps = None
            except Exception:
                messagebox.showerror("Erro", "Formato retornado inesperado.")
                return

        # exibe resultados resumidos
        self.texto_resultado.insert(tk.END, f"=== MÉTODO: {metodo_nome} ===\n\n")
        self.texto_resultado.insert(tk.END, f"STATUS: {status}\n\n")
        if x is not None:
            try:
                sol_arr = np.array(x, dtype=float).reshape(-1)
                self.texto_resultado.insert(tk.END, "SOLUÇÃO (x):\n")
                formatted = "\n".join([f"x[{i}] = {sol_arr[i]:.12g}" for i in range(len(sol_arr))])
                self.texto_resultado.insert(tk.END, formatted + "\n\n")
            except Exception:
                self.texto_resultado.insert(tk.END, f"{x}\n\n")

        self.texto_resultado.insert(tk.END, f"Tempo de execução: {tempo:.6f} s\n\n")

        # tenta calcular resíduo ||Ax - b||_2
        try:
            if x is not None:
                Ax = np.dot(self.A, np.array(x).reshape(-1))
                residuo = np.linalg.norm(Ax - self.b)
                self.texto_resultado.insert(tk.END, f"Resíduo ||Ax - b||_2 = {residuo:.6e}\n")
        except Exception:
            pass

        # se houver passos detalhados, mostra na aba de passos
        if steps:
            self.texto_passos.insert(tk.END, f"=== Passos para {metodo_nome} ===\n\n")
            if "actions" in steps and steps["actions"]:
                self.texto_passos.insert(tk.END, "Ações:\n")
                for act in steps["actions"]:
                    self.texto_passos.insert(tk.END, f"- {act}\n")
                self.texto_passos.insert(tk.END, "\n")
            if "matrizes" in steps and steps["matrizes"]:
                self.texto_passos.insert(tk.END, "Matrizes em etapas:\n")
                for label, mat in steps["matrizes"]:
                    self.texto_passos.insert(tk.END, f"{label}:\n{np.array2string(mat, precision=6, floatmode='maxprec')}\n\n")
            if "L" in steps and steps["L"] is not None:
                self.texto_passos.insert(tk.END, f"Matriz L:\n{np.array2string(steps['L'], precision=6)}\n\n")
            if "U" in steps and steps["U"] is not None:
                self.texto_passos.insert(tk.END, f"Matriz U:\n{np.array2string(steps['U'], precision=6)}\n\n")
            if "col_perm" in steps and steps["col_perm"] is not None:
                self.texto_passos.insert(tk.END, f"Permutação de colunas: {steps['col_perm']}\n\n")
            if "iterations" in steps and steps["iterations"]:
                self.texto_passos.insert(tk.END, "Iterações:\n")
                for k, vec in enumerate(steps["iterations"], start=1):
                    self.texto_passos.insert(tk.END, f"Iter {k}: {np.array2string(vec, precision=6, floatmode='maxprec')}\n")
                self.texto_passos.insert(tk.END, "\n")

    # -------------------------
    # Executa métodos de raízes (usa metodos_raizes)
    # -------------------------
    def run_roots(self):
        # limpa saídas
        self.texto_resultado.delete('1.0', tk.END)
        self.texto_passos.delete('1.0', tk.END)

        method_name = self.root_method_var.get()
        try:
            a = float(self.root_a.get()) if self.root_a.get().strip() != "" else 0.0
            b = float(self.root_b.get()) if self.root_b.get().strip() != "" else 0.0
            x0 = float(self.root_x0.get()) if self.root_x0.get().strip() != "" else 0.0
            x1 = float(self.root_x1.get()) if self.root_x1.get().strip() != "" else 0.0
            tol = float(self.root_tol.get())
            maxit = int(self.root_maxiter.get())
        except Exception as e:
            messagebox.showerror("Erro de entrada", f"Verifique os campos numéricos: {e}")
            return

        # monta estrutura de entrada compatível com metodos_raizes
        # método será ajustado abaixo
        dados = MR.DadosEntrada(1, a, b, x0, x1, tol, maxit)
        mapping = {
            "Bisseção": (1, MR.bissecao),
            "Ponto Fixo": (2, MR.ponto_fixo),
            "Newton-Raphson": (3, MR.newton),
            "Secante": (4, MR.secante),
            "Regula Falsi": (5, MR.regula_falsi)
        }
        metodo_id, func = mapping[method_name]
        dados.metodo = metodo_id

        # captura a saída do método numa string e exibe
        saida_io = io.StringIO()
        try:
            func(dados, saida_io)
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao executar o método de raízes: {e}")
            return

        texto = saida_io.getvalue()
        self.texto_resultado.insert(tk.END, texto)
        if self.var_show_roots_steps.get():
            self.texto_passos.insert(tk.END, texto)

        # tenta salvar em resultado.txt (opcional)
        try:
            with open("resultado.txt", "w", encoding="utf-8") as f:
                f.write(texto)
        except Exception:
            pass

    # -------------------------
    # Salvar, limpar
    # -------------------------
    def salvar_resultado(self):
        texto = self.texto_resultado.get("1.0", tk.END)
        if texto.strip() == "":
            messagebox.showinfo("Salvar", "Nada para salvar.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", ".txt")])
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(texto)
            messagebox.showinfo("Salvar", f"Resultado salvo em {path}")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao salvar: {e}")

    def limpar(self):
        # reseta tudo para o estado inicial
        self.A = None
        self.b = None
        self.lbl_status.config(text="Nenhum arquivo carregado.")
        self.texto_resultado.delete('1.0', tk.END)
        self.texto_passos.delete('1.0', tk.END)
        self.text_a.delete("1.0", tk.END)
        self.text_b.delete("1.0", tk.END)
        self.x0_init.delete(0, tk.END)
        self.x0_init.insert(0, "")
        self.root_a.delete(0, tk.END)
        self.root_b.delete(0, tk.END)
        self.root_x0.delete(0, tk.END)
        self.root_x1.delete(0, tk.END)
        self.root_tol.delete(0, tk.END)
        self.root_tol.insert(0, "1e-6")
        self.root_maxiter.delete(0, tk.END)
        self.root_maxiter.insert(0, "100")
        if hasattr(self, "var_show_steps"):
            self.var_show_steps.set(False)
        if hasattr(self, "var_show_matrices"):
            self.var_show_matrices.set(False)
        if hasattr(self, "var_show_LU"):
            self.var_show_LU.set(False)
        if hasattr(self, "var_show_permutation"):
            self.var_show_permutation.set(False)
        if hasattr(self, "var_show_roots_steps"):
            self.var_show_roots_steps.set(False)


# executa a GUI se o arquivo for rodado diretamente
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
