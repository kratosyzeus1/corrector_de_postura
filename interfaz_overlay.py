"""
=============================================================================
MÓDULO: Interfaz No Disruptiva (Selector Multi-Cámara + Overlay Transparente)
PROYECTO: Asistente Ergonómico Inteligente - Politécnico Grancolombiano
=============================================================================
Implementa:
1. SelectorInicioDialog: Ventana inicial compacta para elegir la cámara cuando
   hay más de una disponible, seleccionar entre Vista 1 (Panel Compacto) y
   Vista 2 (Background / Solo Notificaciones) y activar el modo anti-conflicto.
2. VentanaOverlayNotificaciones (Vista 2): Ejecución en segundo plano sobre
   la pantalla normal del usuario. Muestra únicamente:
   - Una mini-píldora flotante semitransparente (esquina superior derecha) con
     estado postural, score, cámara activa y accesos rápidos.
   - Notificaciones flotantes compactas tipo Toast (esquina inferior derecha)
     que aparecen solo cuando hay alerta postural, sugerencia de pausa o
     notificación de videollamada, sin robar el foco del teclado.
=============================================================================
"""

import sys
import time
from typing import Any, Callable, Dict, List, Optional

try:
    import tkinter as tk
    from tkinter import ttk
    TIENE_TKINTER = True
except Exception:
    TIENE_TKINTER = False

from gestor_camara import InfoCamara


def mostrar_selector_inicio_consola(
    camaras: List[InfoCamara],
    vista_defecto: str = "overlay",
) -> Dict[str, Any]:
    """Respaldo en consola si Tkinter no está disponible."""
    print("\n" + "=" * 66)
    print(" CONFIGURACIÓN INICIAL - ASISTENTE ERGONÓMICO INTELIGENTE")
    print("=" * 66)

    modo_demo = False
    indice_camara = 0

    if len(camaras) == 0:
        print(" [!] No se detectaron cámaras físicas. Se usará Modo Demostración.")
        modo_demo = True
    elif len(camaras) == 1:
        print(f" [OK] 1 cámara detectada: {camaras[0].etiqueta()}")
        indice_camara = camaras[0].indice
    else:
        print(f" [OK] Se detectaron {len(camaras)} cámaras disponibles:")
        for i, cam in enumerate(camaras):
            print(f"   [{i}] {cam.etiqueta()}")
        print(f"   [{len(camaras)}] Modo Demostración Sintético (sin cámara)")
        try:
            sel = input(f" Elige el número de cámara [0]: ").strip()
            opcion = int(sel) if sel else 0
            if opcion == len(camaras):
                modo_demo = True
            elif 0 <= opcion < len(camaras):
                indice_camara = camaras[opcion].indice
        except Exception:
            indice_camara = camaras[0].indice

    print("\n Selecciona la vista inicial:")
    print("   [1] Vista 2 (Recomendada): Segundo plano / Solo notificaciones (Overlay)")
    print("   [2] Vista 1: Panel ergonómico compacto (Avatar sin espejo de cámara)")
    try:
        sel_v = input(" Opción [1]: ").strip()
        vista_inicial = "compacta" if sel_v == "2" else vista_defecto
    except Exception:
        vista_inicial = vista_defecto

    return {
        "iniciar": True,
        "modo_demo": modo_demo,
        "indice_camara": indice_camara,
        "vista_inicial": vista_inicial,
        "anti_conflicto": True,
    }


def mostrar_selector_inicio(
    camaras: List[InfoCamara],
    vista_defecto: str = "overlay",
    auto_cerrar_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Muestra un diálogo visual ligero para elegir cámara (cuando hay múltiples),
    vista inicial (Overlay transparente vs Panel compacto) y protección anti-conflicto.
    """
    if not TIENE_TKINTER:
        return mostrar_selector_inicio_consola(camaras, vista_defecto)

    resultado: Dict[str, Any] = {
        "iniciar": False,
        "modo_demo": len(camaras) == 0,
        "indice_camara": camaras[0].indice if camaras else 0,
        "vista_inicial": vista_defecto,
        "anti_conflicto": True,
    }

    root = tk.Tk()
    root.title("PostuFix | Configuración de Inicio")
    root.configure(bg="#111622")
    root.resizable(False, False)

    ancho_v, alto_v = 470, 405
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f"{ancho_v}x{alto_v}+{(sw - ancho_v) // 2}+{(sh - alto_v) // 2}")

    # Estilos ttk oscuros
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure(
        "Dark.TCombobox",
        fieldbackground="#1b2333",
        background="#253047",
        foreground="#f1f5f9",
        arrowcolor="#38bdf8",
        borderwidth=0,
    )

    # Cabecera
    header = tk.Frame(root, bg="#161d2d", padx=18, pady=12)
    header.pack(fill="x")
    tk.Label(
        header,
        text="Asistente Ergonómico Inteligente",
        bg="#161d2d",
        fg="#f8fafc",
        font=("Segoe UI", 12, "bold"),
        anchor="w",
    ).pack(fill="x")
    tk.Label(
        header,
        text="Selecciona la cámara de lectura y el modo de interfaz menos disruptivo",
        bg="#161d2d",
        fg="#94a3b8",
        font=("Segoe UI", 8),
        anchor="w",
    ).pack(fill="x", pady=(2, 0))

    cuerpo = tk.Frame(root, bg="#111622", padx=18, pady=14)
    cuerpo.pack(fill="both", expand=True)

    # 1. Selector de cámara
    if len(camaras) > 1:
        sub_cam = f"Se detectaron {len(camaras)} cámaras — elige sobre cuál realizar la lectura:"
    elif len(camaras) == 1:
        sub_cam = "1 cámara detectada (puedes cambiarla luego con la tecla [V]):"
    else:
        sub_cam = "No se detectaron cámaras físicas — se iniciará en Modo Demostración:"

    tk.Label(
        cuerpo,
        text="1. Fuente de cámara",
        bg="#111622",
        fg="#38bdf8",
        font=("Segoe UI", 9, "bold"),
        anchor="w",
    ).pack(fill="x")
    tk.Label(
        cuerpo,
        text=sub_cam,
        bg="#111622",
        fg="#cbd5e1",
        font=("Segoe UI", 8),
        anchor="w",
    ).pack(fill="x", pady=(1, 6))

    opciones_cam = [c.etiqueta() for c in camaras]
    opciones_cam.append("Modo Demostración Sintético (sin cámara)")
    var_cam = tk.StringVar(value=opciones_cam[0])

    combo_cam = ttk.Combobox(
        cuerpo,
        textvariable=var_cam,
        values=opciones_cam,
        state="readonly",
        style="Dark.TCombobox",
        font=("Segoe UI", 9),
    )
    combo_cam.pack(fill="x", pady=(0, 12))

    # 2. Selección de Vista
    tk.Label(
        cuerpo,
        text="2. Modo de interfaz (alternable en cualquier momento)",
        bg="#111622",
        fg="#38bdf8",
        font=("Segoe UI", 9, "bold"),
        anchor="w",
    ).pack(fill="x", pady=(0, 5))

    var_vista = tk.StringVar(value=vista_defecto)

    frame_v2 = tk.Frame(cuerpo, bg="#182032", padx=10, pady=7, highlightbackground="#25324a", highlightthickness=1)
    frame_v2.pack(fill="x", pady=(0, 6))
    tk.Radiobutton(
        frame_v2,
        text="Vista 2 (Recomendada): Segundo plano / Solo notificaciones",
        variable=var_vista,
        value="overlay",
        bg="#182032",
        fg="#f8fafc",
        selectcolor="#111622",
        activebackground="#182032",
        activeforeground="#38bdf8",
        font=("Segoe UI", 9, "bold"),
        anchor="w",
    ).pack(fill="x")
    tk.Label(
        frame_v2,
        text="Tu pantalla permanece 100% libre; solo aparecen alertas flotantes discretas.",
        bg="#182032",
        fg="#94a3b8",
        font=("Segoe UI", 8),
        anchor="w",
    ).pack(fill="x", padx=(22, 0))

    frame_v1 = tk.Frame(cuerpo, bg="#182032", padx=10, pady=7, highlightbackground="#25324a", highlightthickness=1)
    frame_v1.pack(fill="x", pady=(0, 10))
    tk.Radiobutton(
        frame_v1,
        text="Vista 1: Cámara en vivo con paneles y mensajes pequeños",
        variable=var_vista,
        value="compacta",
        bg="#182032",
        fg="#f8fafc",
        selectcolor="#111622",
        activebackground="#182032",
        activeforeground="#38bdf8",
        font=("Segoe UI", 9, "bold"),
        anchor="w",
    ).pack(fill="x")
    tk.Label(
        frame_v1,
        text="Muestra lo que ve la cámara con HUD compacto y notificaciones discretas.",
        bg="#182032",
        fg="#94a3b8",
        font=("Segoe UI", 8),
        anchor="w",
    ).pack(fill="x", padx=(22, 0))

    # 3. Anti-conflicto de videollamadas
    var_anti = tk.BooleanVar(value=True)
    tk.Checkbutton(
        cuerpo,
        text="Protección de videollamadas: ceder cámara automáticamente (Zoom, Teams, Meet)",
        variable=var_anti,
        bg="#111622",
        fg="#cbd5e1",
        selectcolor="#1b2333",
        activebackground="#111622",
        activeforeground="#f8fafc",
        font=("Segoe UI", 8),
        anchor="w",
    ).pack(fill="x", pady=(2, 12))

    # Botones de acción
    footer = tk.Frame(cuerpo, bg="#111622")
    footer.pack(fill="x")

    def on_iniciar():
        idx_sel = combo_cam.current()
        if idx_sel < 0 or idx_sel >= len(camaras):
            resultado["modo_demo"] = True
            resultado["indice_camara"] = 0
        else:
            resultado["modo_demo"] = False
            resultado["indice_camara"] = camaras[idx_sel].indice

        resultado["vista_inicial"] = var_vista.get()
        resultado["anti_conflicto"] = bool(var_anti.get())
        resultado["iniciar"] = True
        root.destroy()

    def on_cancelar():
        resultado["iniciar"] = False
        root.destroy()

    tk.Button(
        footer,
        text="Cancelar",
        command=on_cancelar,
        bg="#1e293b",
        fg="#cbd5e1",
        activebackground="#334155",
        activeforeground="#ffffff",
        relief="flat",
        font=("Segoe UI", 9),
        padx=14,
        pady=6,
        cursor="hand2",
    ).pack(side="left")

    tk.Button(
        footer,
        text="Iniciar Monitoreo  ▸",
        command=on_iniciar,
        bg="#0284c7",
        fg="#ffffff",
        activebackground="#0369a1",
        activeforeground="#ffffff",
        relief="flat",
        font=("Segoe UI", 9, "bold"),
        padx=18,
        pady=6,
        cursor="hand2",
    ).pack(side="right")

    root.protocol("WM_DELETE_WINDOW", on_cancelar)
    if auto_cerrar_ms is not None:
        root.after(auto_cerrar_ms, on_iniciar)
    root.mainloop()
    return resultado


class VentanaOverlayNotificaciones:
    """
    Vista 2: Ejecución en segundo plano con ventana transparente / solo notificaciones.

    Mantiene la pantalla del usuario completamente libre:
    - Mini-píldora flotante en la esquina superior derecha (arrastrable, semitransparente)
      con el indicador de postura y controles rápidos.
    - Tarjeta Toast discreta en la esquina inferior derecha que solo aparece cuando
      hay una alerta postural, pausa sugerida o pausa automática por videollamada.
    """

    def __init__(
        self,
        obtener_estado: Callable[[], Dict[str, Any]],
        on_calibrar: Callable[[], None],
        on_cambiar_camara: Callable[[], None],
        on_alternar_pausa: Callable[[], None],
        on_alternar_demo: Callable[[], None],
        on_evaluar_kmeans: Callable[[], None],
    ):
        self.obtener_estado = obtener_estado
        self.on_calibrar = on_calibrar
        self.on_cambiar_camara = on_cambiar_camara
        self.on_alternar_pausa = on_alternar_pausa
        self.on_alternar_demo = on_alternar_demo
        self.on_evaluar_kmeans = on_evaluar_kmeans

        self.accion_salida: str = "salir"  # "compacta" o "salir"
        self._silenciado_hasta: float = 0.0
        self._toast_visible: bool = False
        self._drag_x: int = 0
        self._drag_y: int = 0
        self._ultima_notif_texto: str = ""
        self._notif_temporal_texto: str = ""
        self._notif_temporal_titulo: str = ""
        self._notif_temporal_color: str = "#38bdf8"
        self._notif_temporal_hasta: float = 0.0
        self._after_id: Optional[str] = None
        self._cerrando: bool = False

    def _mostrar_toast_temporal(self, titulo: str, mensaje: str, color: str = "#38bdf8", duracion: float = 3.0):
        self._notif_temporal_titulo = titulo
        self._notif_temporal_texto = mensaje
        self._notif_temporal_color = color
        self._notif_temporal_hasta = time.time() + duracion

    def ejecutar(self, auto_cerrar_ms: Optional[int] = None) -> str:
        """
        Ejecuta el bucle de la Vista 2 (Overlay transparente / solo notificaciones)
        en el hilo principal.
        Retorna: "compacta" si el usuario pidió abrir Vista 1, o "salir" si cerró la app.
        """
        if not TIENE_TKINTER:
            return "compacta"
        self._cerrando = False
        self._after_id = None

        self.root = tk.Tk()
        self.root.title("PostuFix Overlay")
        self.root.overrideredirect(True)
        self.root.configure(bg="#0f172a")

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()

        # --- 1. Mini-píldora flotante (parte superior central, no choca con botones de ventanas) ---
        pill_w, pill_h = 370, 38
        pill_x = max(20, (sw - pill_w) // 2)
        pill_y = 14
        self.root.geometry(f"{pill_w}x{pill_h}+{pill_x}+{pill_y}")

        borde_pill = tk.Frame(self.root, bg="#38bdf8", padx=1, pady=1)
        borde_pill.pack(fill="both", expand=True)

        cont_pill = tk.Frame(borde_pill, bg="#0f172a", padx=10, pady=4)
        cont_pill.pack(fill="both", expand=True)

        # Arrastre de la mini-píldora a cualquier lugar de la pantalla
        def start_drag(ev):
            self._drag_x = ev.x
            self._drag_y = ev.y

        def do_drag(ev):
            nx = self.root.winfo_x() + (ev.x - self._drag_x)
            ny = self.root.winfo_y() + (ev.y - self._drag_y)
            self.root.geometry(f"+{nx}+{ny}")

        self.lbl_dot = tk.Label(cont_pill, text="●", bg="#0f172a", fg="#22c55e", font=("Segoe UI", 11, "bold"))
        self.lbl_dot.pack(side="left", padx=(0, 5))
        self.lbl_dot.bind("<Button-1>", start_drag)
        self.lbl_dot.bind("<B1-Motion>", do_drag)

        self.lbl_estado = tk.Label(
            cont_pill,
            text="100% · C0 · 0MB",
            bg="#0f172a",
            fg="#f1f5f9",
            font=("Segoe UI", 9, "bold"),
        )
        self.lbl_estado.pack(side="left", padx=(0, 8))
        self.lbl_estado.bind("<Button-1>", start_drag)
        self.lbl_estado.bind("<B1-Motion>", do_drag)

        def crear_btn_mini(texto: str, comando, fg: str = "#cbd5e1", bg: str = "#1e293b"):
            btn = tk.Button(
                cont_pill,
                text=texto,
                command=comando,
                bg=bg,
                fg=fg,
                activebackground="#334155",
                activeforeground="#ffffff",
                relief="flat",
                borderwidth=0,
                font=("Segoe UI", 8, "bold"),
                padx=6,
                pady=2,
                cursor="hand2",
            )
            btn.pack(side="right", padx=(4, 0))
            return btn

        crear_btn_mini("✕", self._salir, fg="#fca5a5", bg="#3b1622")
        self.btn_pausa = crear_btn_mini("⏸", self._clic_pausa, fg="#fde68a")
        crear_btn_mini("Calib", self._clic_calibrar, fg="#86efac")
        crear_btn_mini("Cam", self._clic_camara, fg="#93c5fd")
        crear_btn_mini("Vista 1", self._ir_vista_compacta, fg="#38bdf8")

        # Asegurar visibilidad en primer plano sin bloquear la pantalla
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-alpha", 0.92)
        except Exception:
            pass
        self.root.deiconify()
        self.root.lift()
        self.root.update_idletasks()

        # --- 2. Ventana Toast Flotante independiente (esquina inferior derecha, solo en alertas) ---
        self.toast = tk.Toplevel(self.root)
        self.toast.title("Notificación PostuFix")
        self.toast.overrideredirect(True)
        self.toast.configure(bg="#0f172a")

        toast_w, toast_h = 355, 92
        toast_x = max(20, sw - toast_w - 24)
        toast_y = max(60, sh - toast_h - 70)
        self.toast.geometry(f"{toast_w}x{toast_h}+{toast_x}+{toast_y}")

        self.toast_outer = tk.Frame(self.toast, bg="#38bdf8", padx=1, pady=1)
        self.toast_outer.pack(fill="both", expand=True)

        toast_inner = tk.Frame(self.toast_outer, bg="#0f172a")
        toast_inner.pack(fill="both", expand=True)

        self.toast_accent = tk.Frame(toast_inner, bg="#38bdf8", width=5)
        self.toast_accent.pack(side="left", fill="y")

        toast_body = tk.Frame(toast_inner, bg="#0f172a", padx=11, pady=8)
        toast_body.pack(side="left", fill="both", expand=True)

        top_row = tk.Frame(toast_body, bg="#0f172a")
        top_row.pack(fill="x")

        self.lbl_toast_titulo = tk.Label(
            top_row,
            text="Aviso Ergonómico",
            bg="#0f172a",
            fg="#f8fafc",
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        )
        self.lbl_toast_titulo.pack(side="left", fill="x", expand=True)

        tk.Button(
            top_row,
            text="Posponer 2m",
            command=self._silenciar_temporal,
            bg="#1e293b",
            fg="#94a3b8",
            activebackground="#334155",
            activeforeground="#ffffff",
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 8),
            padx=6,
            pady=1,
            cursor="hand2",
        ).pack(side="right")

        self.lbl_toast_msg = tk.Label(
            toast_body,
            text="",
            bg="#0f172a",
            fg="#e2e8f0",
            font=("Segoe UI", 8),
            anchor="w",
            justify="left",
            wraplength=310,
        )
        self.lbl_toast_msg.pack(fill="x", pady=(3, 2))

        self.lbl_toast_meta = tk.Label(
            toast_body,
            text="",
            bg="#0f172a",
            fg="#94a3b8",
            font=("Segoe UI", 7),
            anchor="w",
        )
        self.lbl_toast_meta.pack(fill="x")

        self.toast.attributes("-topmost", True)
        try:
            self.toast.attributes("-alpha", 0.95)
        except Exception:
            pass
        self.toast.withdraw()
        self._toast_visible = False

        # Confirmación inicial visible durante 5 segundos al entrar a Vista 2
        self._mostrar_toast_temporal(
            "Vista 2: Modo Segundo Plano Activo",
            "Tu pantalla queda libre. Aquí aparecerán las alertas posturales y de videollamada.",
            color="#38bdf8",
            duracion=5.0,
        )

        # Programar el primer refresco ya dentro del bucle de eventos (evita fallos de mapeo)
        self._after_id = self.root.after(120, self._programar_refresco)
        if auto_cerrar_ms is not None:
            self.root.after(auto_cerrar_ms, self._salir)
        self.root.mainloop()
        return self.accion_salida

    def _cerrar_ventana(self, accion: str):
        if self._cerrando:
            return
        self._cerrando = True
        self.accion_salida = accion
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        try:
            self.root.destroy()
        except Exception:
            pass

    def _ir_vista_compacta(self):
        self._cerrar_ventana("compacta")

    def _salir(self):
        self._cerrar_ventana("salir")

    def _clic_pausa(self):
        self.on_alternar_pausa()
        self._mostrar_toast_temporal(
            "Control de Cámara",
            "Estado de cámara actualizado (pausa/reanudación manual).",
            color="#f59e0b",
            duracion=2.5,
        )

    def _clic_calibrar(self):
        self.on_calibrar()
        self._mostrar_toast_temporal(
            "Postura Calibrada",
            "Se guardó tu postura neutra actual como referencia.",
            color="#22c55e",
            duracion=3.0,
        )

    def _clic_camara(self):
        self.on_cambiar_camara()
        self._mostrar_toast_temporal(
            "Cambio de Cámara",
            "Cambiando a la siguiente cámara disponible...",
            color="#38bdf8",
            duracion=2.5,
        )

    def _silenciar_temporal(self):
        self._silenciado_hasta = time.time() + 120.0
        self._notif_temporal_hasta = 0.0
        if self._toast_visible:
            self.toast.withdraw()
            self._toast_visible = False

    def _mostrar_o_actualizar_toast(self, titulo: str, mensaje: str, meta: str, color: str):
        self.toast_outer.configure(bg=color)
        self.toast_accent.configure(bg=color)
        self.lbl_toast_titulo.configure(text=titulo, fg=color if color != "#ef4444" else "#fca5a5")
        self.lbl_toast_msg.configure(text=mensaje)
        self.lbl_toast_meta.configure(text=meta)
        if not self._toast_visible:
            self.toast.deiconify()
            self.toast.lift()
            self.toast.attributes("-topmost", True)
            self.toast.update_idletasks()
            self._toast_visible = True

    def _ocultar_toast(self):
        if self._toast_visible:
            self.toast.withdraw()
            self._toast_visible = False

    def _programar_refresco(self):
        if self._cerrando:
            return
        try:
            st = self.obtener_estado()
        except Exception:
            st = {}

        if st.get("detener_solicitado", False):
            self._cerrar_ventana("salir")
            return

        score = int(st.get("score", 100))
        ram_mb = float(st.get("ram_mb", 0.0))
        estado_cam = st.get("estado_camara", "ACTIVA")
        app_conflicto = st.get("app_conflicto", "")
        alerta_activa = bool(st.get("alerta_activa", False))
        mensaje_alerta = str(st.get("mensaje_alerta", ""))
        diagnostico = str(st.get("diagnostico", ""))
        seg_riesgo = float(st.get("segundos_mala_postura", 0.0))
        ventana_alerta = float(st.get("ventana_alerta", 5.0))
        cluster = str(st.get("cluster", "Baja fatiga"))
        ang_cerv = float(st.get("angulo_cervical", 0.0))
        desn_h = float(st.get("desnivel_hombros", 0.0))
        modo_demo = bool(st.get("modo_demo", False))
        cam_idx = st.get("camara_indice", 0)

        # Actualizar mini-píldora
        etiqueta_fuente = "DEMO" if modo_demo else f"C{cam_idx}"
        if estado_cam == "PAUSADA_POR_CONFLICTO" and not modo_demo:
            dot_color = "#38bdf8"
            texto_pill = f"Llamada · {ram_mb:.0f}MB"
            self.btn_pausa.configure(text="⏸")
        elif estado_cam == "PAUSADA_MANUAL" and not modo_demo:
            dot_color = "#94a3b8"
            texto_pill = f"Pausado · {ram_mb:.0f}MB"
            self.btn_pausa.configure(text="▶")
        elif alerta_activa:
            dot_color = "#ef4444"
            texto_pill = f"{score}% · {etiqueta_fuente} · {ram_mb:.0f}MB"
            self.btn_pausa.configure(text="⏸")
        elif score < 80:
            dot_color = "#f59e0b"
            texto_pill = f"{score}% · {etiqueta_fuente} · {ram_mb:.0f}MB"
            self.btn_pausa.configure(text="⏸")
        else:
            dot_color = "#22c55e"
            texto_pill = f"{score}% · {etiqueta_fuente} · {ram_mb:.0f}MB"
            self.btn_pausa.configure(text="⏸")

        self.lbl_dot.configure(fg=dot_color)
        self.lbl_estado.configure(text=texto_pill)

        # Decidir si mostrar Toast flotante (solo cuando es necesario!)
        ahora = time.time()
        silenciado = ahora < self._silenciado_hasta

        if ahora < self._notif_temporal_hasta:
            self._mostrar_o_actualizar_toast(
                self._notif_temporal_titulo,
                self._notif_temporal_texto,
                f"RAM: {ram_mb:.1f} MB  |  Fatiga K-Means: {cluster}",
                self._notif_temporal_color,
            )
        elif estado_cam == "PAUSADA_POR_CONFLICTO" and not modo_demo:
            self._mostrar_o_actualizar_toast(
                "Cámara cedida a videollamada",
                f"En uso por {app_conflicto or 'otra aplicación'}. Se reanudará al terminar.",
                "Sin conflicto de cámara  |  Monitoreo en pausa segura",
                "#38bdf8",
            )
        elif alerta_activa and not silenciado:
            self._mostrar_o_actualizar_toast(
                "Corrección Postural Sugerida",
                mensaje_alerta or "Endereza tu espalda y alinea la cabeza con la pantalla.",
                f"Cervical: {ang_cerv:.1f}° · Hombros: {desn_h:.1f}° · {cluster}",
                "#ef4444",
            )
        elif seg_riesgo >= 1.5 and not silenciado:
            self._mostrar_o_actualizar_toast(
                f"Aviso Postural ({seg_riesgo:.1f}s / {ventana_alerta:.0f}s)",
                diagnostico.replace("RIESGO: ", "") or "Postura fuera de rango — ajusta tu posición.",
                f"Cervical: {ang_cerv:.1f}° · Hombros: {desn_h:.1f}° · {cluster}",
                "#f59e0b",
            )
        else:
            self._ocultar_toast()

        if not self._cerrando:
            self._after_id = self.root.after(250, self._programar_refresco)
