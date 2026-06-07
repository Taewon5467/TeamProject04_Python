import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import os
import threading
from datetime import date, timedelta


# ── 요일 자동 계산 ────────────────────────────────────────
WEEKDAYS = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]

def date_to_weekday(date_str):
    try:
        d = date.fromisoformat(date_str)
        return WEEKDAYS[d.weekday()]
    except Exception:
        return "월요일"


# ── 메인 GUI ─────────────────────────────────────────────
class LauncherApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🤖 배달 로봇 거점 분석")
        self.resizable(False, False)
        self.configure(bg="#0f172a")

        # 폰트 및 색상 설정
        FONT_LBL  = ("맑은 고딕", 10)
        FONT_HEAD = ("맑은 고딕", 11, "bold")
        FONT_BTN  = ("맑은 고딕", 11, "bold")
        COLOR_BG  = "#0f172a"
        COLOR_FRM = "#1e293b"
        COLOR_ACC = "#38bdf8"
        COLOR_TXT = "#e2e8f0"
        COLOR_SUB = "#94a3b8"

        pad = dict(padx=14, pady=6)

        # ── 헤더 ─────────────────────────────────────────
        hdr = tk.Frame(self, bg=COLOR_BG)
        hdr.pack(fill="x", padx=20, pady=(18, 4))
        tk.Label(hdr, text="🤖  배달 로봇 최적 거점 분석",
                 font=("맑은 고딕", 14, "bold"),
                 bg=COLOR_BG, fg=COLOR_ACC).pack(anchor="w")
        tk.Label(hdr, text="시뮬레이션 조건을 설정하고 분석 시작 버튼을 누르세요.",
                 font=("맑은 고딕", 9), bg=COLOR_BG, fg=COLOR_SUB).pack(anchor="w")

        # ── 입력 프레임 ───────────────────────────────────
        frm = tk.Frame(self, bg=COLOR_FRM, bd=0, relief="flat")
        frm.pack(padx=20, pady=10, fill="x")

        def row(parent, r, label, widget_fn):
            tk.Label(parent, text=label, font=FONT_LBL,
                     bg=COLOR_FRM, fg=COLOR_TXT,
                     anchor="w", width=14).grid(row=r, column=0, **pad, sticky="w")
            w = widget_fn(parent)
            w.grid(row=r, column=1, **pad, sticky="ew")
            return w

        frm.columnconfigure(1, weight=1)

        # 날짜
        today_str = str(date.today() + timedelta(days=1))
        self.var_date = tk.StringVar(value=today_str)
        e_date = row(frm, 0, "📅  날짜",
                     lambda p: tk.Entry(p, textvariable=self.var_date,
                                        font=FONT_LBL, width=16,
                                        bg="#0f172a", fg=COLOR_TXT,
                                        insertbackground=COLOR_TXT,
                                        relief="flat", bd=4))
        # 날짜 변경 시 요일 자동 갱신
        self.var_date.trace_add("write", self._sync_weekday)

        # 요일 (자동, read-only)
        self.var_weekday = tk.StringVar(value=date_to_weekday(today_str))
        row(frm, 1, "📆  요일",
            lambda p: ttk.Combobox(p, textvariable=self.var_weekday,
                                   values=WEEKDAYS, state="readonly",
                                   font=FONT_LBL, width=14))

        # 기온
        self.var_temp = tk.DoubleVar(value=25.0)
        row(frm, 2, "🌡️  기온 (°C)",
            lambda p: tk.Spinbox(p, textvariable=self.var_temp,
                                  from_=-20, to=45, increment=0.5,
                                  format="%.1f", font=FONT_LBL, width=10,
                                  bg="#0f172a", fg=COLOR_TXT,
                                  buttonbackground="#1e293b",
                                  relief="flat", bd=4))

        # 강수량
        self.var_rain = tk.DoubleVar(value=0.0)
        row(frm, 3, "🌧️  강수량 (mm)",
            lambda p: tk.Spinbox(p, textvariable=self.var_rain,
                                  from_=0, to=200, increment=0.5,
                                  format="%.1f", font=FONT_LBL, width=10,
                                  bg="#0f172a", fg=COLOR_TXT,
                                  buttonbackground="#1e293b",
                                  relief="flat", bd=4))

        # 적설
        self.var_snow = tk.DoubleVar(value=0.0)
        row(frm, 4, "❄️  적설 (cm)",
            lambda p: tk.Spinbox(p, textvariable=self.var_snow,
                                  from_=0, to=100, increment=0.5,
                                  format="%.1f", font=FONT_LBL, width=10,
                                  bg="#0f172a", fg=COLOR_TXT,
                                  buttonbackground="#1e293b",
                                  relief="flat", bd=4))

        # 시정
        self.var_vis = tk.DoubleVar(value=2000.0)
        row(frm, 5, "👁️  시정 (m)",
            lambda p: tk.Spinbox(p, textvariable=self.var_vis,
                                  from_=0, to=10000, increment=100,
                                  format="%.0f", font=FONT_LBL, width=10,
                                  bg="#0f172a", fg=COLOR_TXT,
                                  buttonbackground="#1e293b",
                                  relief="flat", bd=4))

        # 공휴일
        self.var_holiday = tk.BooleanVar(value=False)
        row(frm, 6, "🎌  공휴일",
            lambda p: tk.Checkbutton(p, variable=self.var_holiday,
                                      bg=COLOR_FRM, fg=COLOR_TXT,
                                      selectcolor="#0f172a",
                                      activebackground=COLOR_FRM,
                                      relief="flat"))

        # 결과 저장
        self.var_save = tk.BooleanVar(value=True)
        row(frm, 7, "💾  결과 저장",
            lambda p: tk.Checkbutton(p, variable=self.var_save,
                                      bg=COLOR_FRM, fg=COLOR_TXT,
                                      selectcolor="#0f172a",
                                      activebackground=COLOR_FRM,
                                      relief="flat"))

        # ── 실행 버튼 ─────────────────────────────────────
        btn_row = tk.Frame(self, bg=COLOR_BG)
        btn_row.pack(fill="x", padx=20, pady=(0, 6))
        btn_row.columnconfigure(0, weight=3)
        btn_row.columnconfigure(1, weight=1)

        self.btn_run = tk.Button(
            btn_row, text="▶  분석 시작",
            font=FONT_BTN, bg="#0ea5e9", fg="white",
            activebackground="#38bdf8", activeforeground="white",
            relief="flat", bd=0, padx=0, pady=10,
            cursor="hand2", command=self._run
        )
        self.btn_run.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.btn_stop = tk.Button(
            btn_row, text="⏹  서버 종료",
            font=FONT_BTN, bg="#334155", fg="#94a3b8",
            activebackground="#475569", activeforeground="white",
            relief="flat", bd=0, padx=0, pady=10,
            cursor="hand2", command=self._stop, state="disabled"
        )
        self.btn_stop.grid(row=0, column=1, sticky="ew")

        self._proc = None  # 실행 중인 프로세스

        # ── 로그 출력창 ───────────────────────────────────
        log_frm = tk.Frame(self, bg=COLOR_BG)
        log_frm.pack(fill="both", expand=True, padx=20, pady=(0, 16))

        self.log = tk.Text(log_frm, height=10, font=("Consolas", 9),
                           bg="#020617", fg="#a3e635",
                           insertbackground="#a3e635",
                           relief="flat", bd=0, state="disabled",
                           wrap="word")
        sb = ttk.Scrollbar(log_frm, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self._log("준비 완료. 조건을 설정하고 분석 시작 버튼을 누르세요.\n")

        # ── 창 닫기(X) 이벤트 감지 등록 ────────────────
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    # ── 창이 닫힐 때 호출되는 메서드 (서버 동시 종료) ─────────────────
    def _on_closing(self):
        if self._proc and self._proc.poll() is None:
            if messagebox.askokcancel("종료 확인", "분석 서버가 아직 실행 중입니다.\n서버를 종료하고 프로그램을 닫으시겠습니까?"):
                self._stop()  # 백그라운드 프로세스 종료 처리
                self.destroy() # GUI 창 닫기
        else:
            self.destroy()

    # ── 날짜 → 요일 자동 동기화 ──────────────────────────────
    def _sync_weekday(self, *_):
        wd = date_to_weekday(self.var_date.get())
        self.var_weekday.set(wd)

    # ── 로그 출력 ─────────────────────────────────────────
    def _log(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg)
        self.log.see("end")
        self.log.configure(state="disabled")

    # ── 서버 종료 ─────────────────────────────────────────
    def _stop(self):
        if self._proc and self._proc.poll() is None:
            try:
                # 윈도우에서 자식 프로세스(visualization 서버)까지 트리 구조로 
                # 통째로 강제 종료시키는 가장 확실한 명령어 (wmic / taskkill 트리 구조 활용)
                pid = self._proc.pid
                os.system(f"taskkill /F /T /PID {pid}")
                
                self._proc.wait(timeout=2)
            except Exception as e:
                pass
                
        # 💡 [혹시 모를 대책] 포트 8000번이 강제로 안 풀렸을 때를 대비해 한 번 더 저격 종료
        try:
            os.system("taskkill /F /IM python.exe /FI \"WINDOWTITLE eq 🌐_delivery_robot_server_🌐\"")
        except Exception:
            pass

        self._proc = None
        self.btn_stop.configure(state="disabled", fg="#94a3b8", bg="#334155")
        self.btn_run.configure(state="normal", text="▶  분석 시작")
        self._log("⏹  서버가 완전히 강제 종료되었습니다. (유령 프로세스 박멸 완료)\n")
        
    def _handle_success(self):
        # 이미 완료 처리가 되었다면 중복 실행 방지
        if self.btn_run["text"] == "▶  분석 시작" and self.btn_stop["state"] == "normal":
            return
            
        self._log("\n✅ 분석 완료! 브라우저에서 지도를 확인하세요.\n")
        self._log("🌐 서버가 실행 중입니다. 지도 확인 후 [서버 종료] 버튼을 누르세요.\n")
        self.btn_stop.configure(state="normal", fg="white", bg="#ef4444")
        self.btn_run.configure(state="normal", text="▶  분석 시작")

    # ── 분석 실행 ─────────────────────────────────────────
    def _run(self):
        # 입력값 검증
        date_str = self.var_date.get().strip()
        try:
            date.fromisoformat(date_str)
        except ValueError:
            messagebox.showerror("입력 오류", "날짜 형식이 올바르지 않습니다.\n예: 2026-06-13")
            return

        self.btn_run.configure(state="disabled", text="⏳  분석 중...")
        self._log(f"\n{'─'*40}\n")
        self._log(f"📅 날짜: {date_str} ({self.var_weekday.get()})\n")
        self._log(f"🌡️  기온: {self.var_temp.get()}°C  "
                  f"🌧️ 강수: {self.var_rain.get()}mm  "
                  f"❄️ 적설: {self.var_snow.get()}cm  "
                  f"👁️ 시정: {self.var_vis.get()}m\n")

        script_dir = os.path.dirname(os.path.abspath(__file__))

        cmd = [
            sys.executable, "-u", "main.py",
            "--date",    date_str,
            "--weekday", self.var_weekday.get(),
            "--temp",    str(self.var_temp.get()),
            "--rain",    str(self.var_rain.get()),
            "--snow",    str(self.var_snow.get()),
            "--vis",     str(self.var_vis.get()),
        ]
        if self.var_holiday.get():
            cmd.append("--holiday")
        if not self.var_save.get():
            cmd.append("--no-save")

        def _worker():
            try:
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                env["PYTHONUTF8"] = "1"

                self._proc = subprocess.Popen(
                    cmd, cwd=script_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace",
                    stdin=subprocess.DEVNULL,
                    env=env,
                )
                
                for line in self._proc.stdout:
                    self.after(0, self._log, line)
                    
                    # 🛠️ [수정] 실제 출력되는 단어("서버 실행", "로컬 서버", "localhost")와 일치하도록 조건 확장
                    if "서버" in line or "localhost" in line or "127.0.0.1" in line:
                        self.after(200, lambda: self._handle_success())

                self._proc.wait()

                if self._proc and self._proc.returncode == 0:
                    self.after(0, self._handle_success)
                else:
                    if self._proc and self._proc.returncode != 0 and self.btn_run["text"] == "⏳  분석 중...":
                        self.after(0, self._log, f"\n❌ 오류 발생 (코드: {self._proc.returncode})\n")
                        
            except Exception as e:
                self.after(0, self._log, f"\n❌ 실행 오류: {e}\n")
            finally:
                if self._proc is None or self._proc.poll() is not None:
                    if self.btn_run["text"] == "⏳  분석 중...":
                        self.after(0, self.btn_run.configure, {"state": "normal", "text": "▶  분석 시작"})

        threading.Thread(target=_worker, daemon=True).start()


if __name__ == "__main__":
    app = LauncherApp()
    app.mainloop()