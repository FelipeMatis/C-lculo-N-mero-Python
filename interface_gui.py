import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import os
import io
import datetime

from metodos_lineares import METODOS as METODOS_LIN
import metodos_raizes as MR  # funções: bissecao, newton, secante, etc.


# ---------------- utilitários ----------------
def ensure_logs_dir():
    os.makedirs("logs", exist_ok=True)

def timestamp_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------- App ----------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Trabalho - Métodos do Capítulo 3")
        self.root.geometry("1200x760")
        self.root.minsize(880, 650)
        self.A = None
        self.b = None

        ensure_logs_dir()
        self._setup_style()

        # ======= Estrutura com rolagem principal (corrigida para reduzir artefatos de render) =======
        main_frame = tk.Frame(self.root, bg="#f4f6fb")
        main_frame.pack(fill="both", expand=True)

        canvas = tk.Canvas(main_frame, background="#f4f6fb", highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=scrollbar.set)

        # frame interno com background explícito para evitar 'faixa branca'
        self.scrollable_frame = tk.Frame(canvas, bg="#f4f6fb")

        # cria a window dentro do canvas e guarda o id para ajustar largura depois
        self._canvas_window_id = canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # atualiza scrollregion quando o conteúdo mudar
        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        self.scrollable_frame.bind("<Configure>", _on_frame_configure)

        # Quando o canvas muda de tamanho, define a largura da window interna como inteiro
        def _on_canvas_configure(event):
            try:
                w = int(event.width)  # largura inteira evita subpixel
                canvas.itemconfigure(self._canvas_window_id, width=w)
            except Exception:
                pass
        canvas.bind("<Configure>", _on_canvas_configure)

        # Função de scroll que força uma repintura leve após mover
        def _on_mousewheel(event):
            # calcula delta inteiro (unidades)
            if hasattr(event, "delta"):
                # Windows / MacOS normalmente usam delta múltiplo de 120
                try:
                    delta = int(-1 * (event.delta / 120))
                except Exception:
                    delta = -1 if event.delta > 0 else 1
            else:
                # X11 wheel events (Button-4/5)
                delta = -1 if event.num == 4 else 1

            canvas.yview_scroll(delta, "units")

            # força repintura leve para reduzir artefatos visuais
            try:
                canvas.update_idletasks()
                self.scrollable_frame.update_idletasks()
                canvas.update()
            except Exception:
                pass

        # Bind/unbind do mousewheel somente enquanto o mouse está sobre o canvas
        def _bind_mousewheel(event):
            # Windows / MacOS
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            # X11
            canvas.bind_all("<Button-4>", _on_mousewheel)
            canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_mousewheel(event):
            try:
                canvas.unbind_all("<MouseWheel>")
            except Exception:
                pass
            try:
                canvas.unbind_all("<Button-4>")
                canvas.unbind_all("<Button-5>")
            except Exception:
                pass

        canvas.bind("<Enter>", _bind_mousewheel)
        canvas.bind("<Leave>", _unbind_mousewheel)

        # guarda canvas para uso futuro
        self._canvas = canvas

        # Monta UI dentro do frame rolável (chamada única e correta)
        self._build_ui()

    # ---------------- estilo ----------------
    def _setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        bg = "#f4f6fb"
        accent = "#1a237e"
        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI Semibold", 13), foreground=accent, background=bg)
        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        style.configure("TEntry", padding=4)
        style.configure("TCombobox", padding=4)
        self.root.configure(bg=bg)

    # ---------------- layout helpers ----------------
    def _make_section(self, parent, title):
        outer = tk.Frame(parent, bg="#f4f6fb")
        outer.pack(fill="x", pady=(10, 6))

        title_label = ttk.Label(outer, text=title, style="Header.TLabel")
        title_label.pack(anchor="w", padx=6, pady=(0, 4))

        shadow = tk.Frame(outer, bg="#e6e9ef")
        shadow.pack(fill="x", padx=6)
        frame = tk.Frame(shadow, bg="#ffffff", padx=10, pady=10)
        frame.pack(fill="x", padx=2, pady=2)
        return frame

    # ---------------- interface ----------------
    def _build_ui(self):
        main = ttk.Frame(self.scrollable_frame, padding=12)
        main.pack(fill="both", expand=True)

        title = ttk.Label(main, text="Trabalho — Métodos Numéricos", style="Header.TLabel")
        title.pack(anchor="w", pady=(0, 8))

        # Seções principais
        load_frame = self._make_section(main, "Carregar Sistema (Matriz A e Vetor b)")
        self._build_load_section(load_frame)

        linear_frame = self._make_section(main, "Métodos Lineares")
        self._build_linear_section(linear_frame)

        params_frame = self._make_section(main, "Parâmetros (Métodos Iterativos)")
        self._build_params_section(params_frame)

        roots_frame = self._make_section(main, "Métodos para Raízes")
        self._build_roots_section(roots_frame)

        # ======== Barra de ações ========
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill="x", pady=8)
        for i in range(5):
            btn_frame.columnconfigure(i, weight=1)

        ttk.Button(btn_frame, text="Resolver Método Linear", command=self.resolver_linear).grid(row=0, column=0, padx=6, sticky="ew")
        ttk.Button(btn_frame, text="Resolver Raiz", command=self.run_roots).grid(row=0, column=1, padx=6, sticky="ew")
        ttk.Button(btn_frame, text="Limpar Campos", command=self.limpar).grid(row=0, column=2, padx=6, sticky="ew")
        ttk.Button(btn_frame, text="Limpar Saída", command=self.limpar_saida).grid(row=0, column=3, padx=6, sticky="ew")
        ttk.Button(btn_frame, text="Salvar Resultado (.txt)", command=self.salvar_resultado).grid(row=0, column=4, padx=6, sticky="ew")

        # ======== Área de resultados ========
        self.result_nb = ttk.Notebook(main)
        self.result_nb.pack(fill="both", expand=True, pady=(6, 0))

        # Aba de resultados
        result_frame = ttk.Frame(self.result_nb)
        self.result_nb.add(result_frame, text="Resultados")
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)

        self.texto_resultado = tk.Text(result_frame, wrap=tk.NONE, font=("Consolas", 10), bd=1, relief="solid")
        self.texto_resultado.grid(row=0, column=0, sticky="nsew")

        vsb = ttk.Scrollbar(result_frame, orient="vertical", command=self.texto_resultado.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        self.texto_resultado.configure(yscrollcommand=vsb.set)

        hsb = ttk.Scrollbar(result_frame, orient="horizontal", command=self.texto_resultado.xview)
        hsb.grid(row=1, column=0, sticky="ew")
        self.texto_resultado.configure(xscrollcommand=hsb.set)

        # Aba de passos
        steps_frame = ttk.Frame(self.result_nb)
        self.result_nb.add(steps_frame, text="Passos / Saída detalhada")
        steps_frame.columnconfigure(0, weight=1)
        steps_frame.rowconfigure(0, weight=1)

        self.texto_passos = tk.Text(steps_frame, wrap=tk.NONE, font=("Consolas", 10), bd=1, relief="solid")
        self.texto_passos.grid(row=0, column=0, sticky="nsew")

        vsb2 = ttk.Scrollbar(steps_frame, orient="vertical", command=self.texto_passos.yview)
        vsb2.grid(row=0, column=1, sticky="ns")
        self.texto_passos.configure(yscrollcommand=vsb2.set)

        hsb2 = ttk.Scrollbar(steps_frame, orient="horizontal", command=self.texto_passos.xview)
        hsb2.grid(row=1, column=0, sticky="ew")
        self.texto_passos.configure(xscrollcommand=hsb2.set)

        # Chamada inicial para adaptar opções
        self._on_metodo_change()

    # ---------------- Build sections ----------------
    def _build_load_section(self, frame):
        btn_row = ttk.Frame(frame)
        btn_row.pack(fill="x", pady=(0,6))
        ttk.Button(btn_row, text="Carregar arquivo (A|b)", command=self.load_ab_file).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Carregar A (apenas coeficientes)", command=self.load_a_file).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Carregar b (vetor)", command=self.load_b_file).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Carregar do texto", command=self.load_from_text).pack(side="left", padx=4)

        self.lbl_status = ttk.Label(frame, text="Nenhum arquivo carregado.")
        self.lbl_status.pack(anchor="w", pady=(4,4))

        paste_frame = ttk.Frame(frame)
        paste_frame.pack(fill="x")
        left = ttk.Frame(paste_frame)
        left.pack(side="left", fill="both", expand=True, padx=(0,10))
        ttk.Label(left, text="Colar A (linha por linha):").pack(anchor="w")
        self.text_a = tk.Text(left, height=6, font=("Consolas",10), bd=1, relief="solid")
        self.text_a.pack(fill="both", expand=True, pady=6)
        right = ttk.Frame(paste_frame, width=180)
        right.pack(side="left", fill="y")
        ttk.Label(right, text="Colar b (linha ou coluna):").pack(anchor="w")
        self.text_b = tk.Text(right, height=6, width=18, font=("Consolas",10), bd=1, relief="solid")
        self.text_b.pack(fill="both", expand=False, pady=6)

    def _build_linear_section(self, frame):
        ttk.Label(frame, text="Escolha o método:").pack(anchor="w")
        # evita erro caso METODOS_LIN esteja vazio
        default_method = list(METODOS_LIN.keys())[0] if len(METODOS_LIN) > 0 else ""
        self.metodo_selecionado = tk.StringVar(value=default_method)
        metodo_combo = ttk.Combobox(frame, textvariable=self.metodo_selecionado, values=list(METODOS_LIN.keys()), state="readonly")
        metodo_combo.pack(fill="x", pady=6)
        metodo_combo.bind("<<ComboboxSelected>>", lambda e: self._on_metodo_change())

        self.options_frame = ttk.Frame(frame)
        self.options_frame.pack(fill="x", pady=4)
        self._build_linear_options()

    def _build_params_section(self, frame):
        grid = ttk.Frame(frame)
        grid.pack(fill="x")
        ttk.Label(grid, text="Tolerância:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.tol = ttk.Entry(grid, width=12)
        self.tol.insert(0, "1e-8")
        self.tol.grid(row=0, column=1, padx=6, pady=4)
        ttk.Label(grid, text="Máx. iterações:").grid(row=0, column=2, sticky="w", padx=6)
        self.max_iter = ttk.Entry(grid, width=12)
        self.max_iter.insert(0, "1000")
        self.max_iter.grid(row=0, column=3, padx=6, pady=4)
        ttk.Label(grid, text="Chute inicial x0 (separado por vírgula):").grid(row=1, column=0, columnspan=2, sticky="w", padx=6)
        self.x0_init = ttk.Entry(grid)
        self.x0_init.grid(row=1, column=2, columnspan=2, sticky="ew", padx=6, pady=(0,6))

    def _build_roots_section(self, frame):
        row = ttk.Frame(frame)
        row.pack(fill="x")
        ttk.Label(row, text="Método:").pack(side="left")
        self.root_method_var = tk.StringVar(value="Bisseção")
        root_methods = ["Bisseção", "Ponto Fixo", "Newton-Raphson", "Secante", "Regula Falsi"]
        root_combo = ttk.Combobox(row, textvariable=self.root_method_var, values=root_methods, state="readonly")
        root_combo.pack(side="left", padx=8)
        root_combo.bind("<<ComboboxSelected>>", lambda e: self._on_metodo_change())

        coord_frame = ttk.Frame(frame)
        coord_frame.pack(fill="x", pady=8)
        ttk.Label(coord_frame, text="a:").grid(row=0, column=0, sticky="e")
        self.root_a = ttk.Entry(coord_frame, width=12)
        self.root_a.grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(coord_frame, text="b:").grid(row=0, column=2, sticky="e")
        self.root_b = ttk.Entry(coord_frame, width=12)
        self.root_b.grid(row=0, column=3, sticky="w", padx=6)
        ttk.Label(coord_frame, text="x0:").grid(row=1, column=0, sticky="e")
        self.root_x0 = ttk.Entry(coord_frame, width=12)
        self.root_x0.grid(row=1, column=1, sticky="w", padx=6)
        ttk.Label(coord_frame, text="x1:").grid(row=1, column=2, sticky="e")
        self.root_x1 = ttk.Entry(coord_frame, width=12)
        self.root_x1.grid(row=1, column=3, sticky="w", padx=6)
        ttk.Label(coord_frame, text="Tolerância:").grid(row=2, column=0, sticky="e")
        self.root_tol = ttk.Entry(coord_frame, width=12)
        self.root_tol.insert(0, "1e-6")
        self.root_tol.grid(row=2, column=1, sticky="w", padx=6)
        ttk.Label(coord_frame, text="Max iterações:").grid(row=2, column=2, sticky="e")
        self.root_maxiter = ttk.Entry(coord_frame, width=12)
        self.root_maxiter.insert(0, "100")
        self.root_maxiter.grid(row=2, column=3, sticky="w", padx=6)
        self.var_show_roots_steps = tk.BooleanVar(value=False)
        ttk.Checkbutton(coord_frame, text="Mostrar passos", variable=self.var_show_roots_steps).grid(row=3, column=0, columnspan=2, sticky="w", padx=6)

    # ---------------- options ----------------
    def _build_linear_options(self):
        for w in self.options_frame.winfo_children():
            w.destroy()
        ttk.Label(self.options_frame, text="Opções específicas do método:").pack(anchor="w")
        self.var_show_steps = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.options_frame, text="Mostrar passos detalhados", variable=self.var_show_steps).pack(anchor="w", padx=6, pady=2)
        self.var_show_matrices = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.options_frame, text="Incluir matrizes em cada passo", variable=self.var_show_matrices).pack(anchor="w", padx=6, pady=2)
        self.var_show_LU = tk.BooleanVar(value=False)
        self.cb_LU = ttk.Checkbutton(self.options_frame, text="Exibir L e U (quando aplicável)", variable=self.var_show_LU)
        self.var_show_permutation = tk.BooleanVar(value=False)
        self.cb_perm = ttk.Checkbutton(self.options_frame, text="Exibir permutações (pivoteamento)", variable=self.var_show_permutation)
        self._on_metodo_change()

    def _on_metodo_change(self):
        metodo = self.metodo_selecionado.get().lower() if hasattr(self, "metodo_selecionado") else ""
        try:
            self.cb_LU.pack_forget()
            self.cb_perm.pack_forget()
        except Exception:
            pass
        if "lu" in metodo or "fatoração lu" in metodo or "fatoracao lu" in metodo:
            try:
                self.cb_LU.pack(anchor="w", padx=6, pady=2)
            except Exception:
                pass
        if "completo" in metodo or "pivoteamento completo" in metodo:
            try:
                self.cb_perm.pack(anchor="w", padx=6, pady=2)
            except Exception:
                pass

    # ---------------- parsing / loading ----------------
    def parse_text_matrix(self, txt):
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip() != ""]
        mat = []
        for ln in lines:
            parts = ln.split()
            mat.append([float(x) for x in parts])
        return np.array(mat, dtype=float)

    def load_ab_file(self):
        path = filedialog.askopenfilename(title="Selecionar arquivo A|b", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
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
                messagebox.showerror("Erro", "Formato inválido para arquivo estendido (A|b).")
                return
            self.A = mat[:, :-1]
            self.b = mat[:, -1].reshape(-1)
            self.lbl_status.config(text=f"Carregado A|b de: {os.path.basename(path)} (A: {self.A.shape}, b: {self.b.shape})")
            self.texto_resultado.insert(tk.END, f"Arquivo '{os.path.basename(path)}' carregado como A|b.\n")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao carregar arquivo: {e}")

    def load_a_file(self):
        path = filedialog.askopenfilename(title="Selecionar arquivo A", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
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
                messagebox.showerror("Erro", "Formato inválido para matriz A. Deve ser quadrada.")
                return
            self.A = mat
            self.lbl_status.config(text=f"Carregado A de: {os.path.basename(path)} (A: {self.A.shape})")
            self.texto_resultado.insert(tk.END, f"Arquivo '{os.path.basename(path)}' carregado como A.\n")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao carregar arquivo A: {e}")

    def load_b_file(self):
        path = filedialog.askopenfilename(title="Selecionar arquivo b", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
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
            self.lbl_status.config(text=f"Carregado b de: {os.path.basename(path)} (b: {self.b.shape})")
            self.texto_resultado.insert(tk.END, f"Arquivo '{os.path.basename(path)}' carregado como b.\n")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao carregar arquivo b: {e}")

    def load_from_text(self):
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

                if b_txt == "" and A.shape[1] == A.shape[0] + 1:
                    self.A = A[:, :-1]
                    self.b = A[:, -1].reshape(-1)
                else:
                    if A.shape[0] != A.shape[1]:
                        messagebox.showerror("Erro", "Se estiver carregando apenas A pelo texto, A deve ser quadrada (n x n).")
                        return
                    self.A = A

            if b_txt != "":
                parts = []
                for ln in b_txt.splitlines():
                    for x in ln.split():
                        parts.append(float(x))
                self.b = np.array(parts, dtype=float).reshape(-1)

            self.lbl_status.config(text=f"Carregado A e/ou b do texto (A: {self.A.shape if self.A is not None else None}, b: {self.b.shape if self.b is not None else None})")
            self.texto_resultado.insert(tk.END, "Dados carregados do texto.\n")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao carregar do texto: {e}")

    def load_entrada_file(self):
        path = filedialog.askopenfilename(title="Selecionar entrada.txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                tokens = f.read().strip().split()
                if len(tokens) < 7:
                    messagebox.showerror("Erro", "entrada.txt precisa ter 7 valores: metodo a b x0 x1 tol maxIter")
                    return
                metodo = int(tokens[0])
                a = tokens[1]
                b = tokens[2]
                x0 = tokens[3]
                x1 = tokens[4]
                tol = tokens[5]
                maxit = tokens[6]

                self.root_method_var.set(["Bisseção", "Ponto Fixo", "Newton-Raphson", "Secante", "Regula Falsi"][metodo-1])
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

    # ----------------- resolver linear -----------------
    def resolver_linear(self):
        self.texto_resultado.delete('1.0', tk.END)
        self.texto_passos.delete('1.0', tk.END)

        if self.A is None or self.b is None:
            messagebox.showerror("Erro", "Por favor, carregue a matriz A e o vetor b.")
            return

        metodo_nome = self.metodo_selecionado.get()
        if metodo_nome not in METODOS_LIN:
            messagebox.showerror("Erro", f"Método '{metodo_nome}' não encontrado em METODOS.")
            return
        metodo_func = METODOS_LIN[metodo_nome]

        show_steps = self.var_show_steps.get() if hasattr(self, "var_show_steps") else False
        show_matrices = self.var_show_matrices.get() if hasattr(self, "var_show_matrices") else False
        show_LU = self.var_show_LU.get() if hasattr(self, "var_show_LU") else False
        show_perm = self.var_show_permutation.get() if hasattr(self, "var_show_permutation") else False

        try:
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
                # call with options if supported by method
                try:
                    sol = metodo_func(self.A, self.b,
                                      return_steps=show_steps,
                                      show_steps_matrix=show_matrices,
                                      show_LU=show_LU,
                                      show_permutation=show_perm)
                except TypeError:
                    # fallback to simpler signature
                    sol = metodo_func(self.A, self.b)
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao executar método linear: {e}")
            return

    # (restante do arquivo continua igual — corte aqui para economia de espaço)
    # Obs: para não quebrar nada, o restante das funções (interpretação do resultado,
    # exibição de passos, métodos de raízes, salvar, limpar) permanecem idênticos ao que você já tinha.
    # Se quiser que eu cole o arquivo completo novamente sem truncar, eu colo na íntegra.

    # ----------------- métodos de raízes -----------------
    def run_roots(self):
        self.texto_resultado.delete('1.0', tk.END)
        self.texto_passos.delete('1.0', tk.END)

        method_name = self.root_method_var.get()
        try:
            a = float(self.root_a.get()) if self.root_a.get().strip() != "" else None
            b = float(self.root_b.get()) if self.root_b.get().strip() != "" else None
            x0 = float(self.root_x0.get()) if self.root_x0.get().strip() != "" else None
            x1 = float(self.root_x1.get()) if self.root_x1.get().strip() != "" else None
            tol = float(self.root_tol.get())
            maxit = int(self.root_maxiter.get())
        except Exception as e:
            messagebox.showerror("Erro de entrada", f"Verifique os campos numéricos: {e}")
            return

        needs = {
            "Bisseção": ("a", "b"),
            "Regula Falsi": ("a", "b"),
            "Secante": ("x0", "x1"),
            "Newton-Raphson": ("x0",),
            "Ponto Fixo": ("x0",),
        }
        required = needs.get(method_name, ())
        provided = {"a": a is not None, "b": b is not None, "x0": x0 is not None, "x1": x1 is not None}
        for r in required:
            if not provided[r]:
                messagebox.showerror("Erro de Entrada", f"Método {method_name} requer campo {r} preenchido.")
                return

        D = MR.DadosEntrada(1, a if a is not None else 0.0, b if b is not None else 0.0,
                             x0 if x0 is not None else 0.0, x1 if x1 is not None else 0.0,
                             tol, maxit)

        mapping = {
            "Bisseção": (1, MR.bissecao),
            "Ponto Fixo": (2, MR.ponto_fixo),
            "Newton-Raphson": (3, MR.newton),
            "Secante": (4, MR.secante),
            "Regula Falsi": (5, MR.regula_falsi)
        }
        if method_name not in mapping:
            messagebox.showerror("Erro", f"Método de raízes '{method_name}' não mapeado.")
            return
        metodo_id, func = mapping[method_name]
        D.metodo = metodo_id

        if method_name == "Ponto Fixo":
            try:
                if hasattr(MR, 'phi'):
                    x_eval = D.x0
                    h = 1e-6
                    phi = MR.phi
                    deriv = (phi(x_eval + h) - phi(x_eval - h)) / (2 * h)
                    if abs(deriv) >= 1.0:
                        self.texto_resultado.insert(tk.END, f"⚠️ Aviso: |phi'(x0)| ≈ {deriv:.6f} >= 1 → ponto fixo pode não convergir.\n\n")
            except Exception:
                pass

        saida_io = io.StringIO()
        try:
            func(D, saida_io)
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao executar o método de raízes: {e}")
            return

        texto = saida_io.getvalue()
        self.texto_resultado.insert(tk.END, texto)
        if self.var_show_roots_steps.get():
            self.texto_passos.insert(tk.END, texto)

        # tenta salvar log automático
        try:
            fn = f"logs/resultado_{timestamp_str()}.txt"
            with open(fn, 'w', encoding='utf-8') as f:
                f.write(texto)
            self.texto_resultado.insert(tk.END, f"\n[Salvo log automático em: {fn}]\n")
        except Exception:
            pass

        self.texto_resultado.see(tk.END)
        self.texto_passos.see(tk.END)

    # ---------------- salvar / limpar ----------------
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
        # limpa tudo exceto arquivos carregados (A e b)
        try:
            self.texto_resultado.delete('1.0', tk.END)
            self.texto_passos.delete('1.0', tk.END)
            self.text_a.delete("1.0", tk.END)
            self.text_b.delete("1.0", tk.END)
        except Exception:
            pass
        try:
            self.x0_init.delete(0, tk.END)
            self.x0_init.insert(0, "")
        except Exception:
            pass
        try:
            self.root_a.delete(0, tk.END)
            self.root_b.delete(0, tk.END)
            self.root_x0.delete(0, tk.END)
            self.root_x1.delete(0, tk.END)
            self.root_tol.delete(0, tk.END)
            self.root_tol.insert(0, "1e-6")
            self.root_maxiter.delete(0, tk.END)
            self.root_maxiter.insert(0, "100")
        except Exception:
            pass
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
        self.lbl_status.config(text="Campos limpos.")

    def limpar_saida(self):
        self.texto_resultado.delete('1.0', tk.END)
        self.texto_passos.delete('1.0', tk.END)
        self.lbl_status.config(text="Saída limpa.")


# ---------------- run ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
