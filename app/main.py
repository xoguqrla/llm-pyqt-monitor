import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = QWidget()
    w.setWindowTitle("공정 데이터 LLM 분석 (PyQt)")
    lay = QVBoxLayout(w)
    lay.addWidget(QLabel("Hello PyQt – 프로젝트 구조가 준비되었습니다."))
    w.resize(800, 500)
    w.show()
    sys.exit(app.exec_())
