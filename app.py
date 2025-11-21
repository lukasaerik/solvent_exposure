import sys, os, time, mplcursors, traceback, tempfile
import numpy as np
from pathlib import Path
from uuid import uuid4

flags = [
    "--disable-logging",
    "--log-level=3",
    "--disable-software-rasterizer",
    "--disable-gpu",               # try remove this if you want to allow GPU; keep if crashes
    "--disable-gpu-compositing",
    "--disable-gpu-sandbox",
    "--disable-accelerated-2d-canvas",
    "--in-process-gpu",              # keep GPU work in main process; prevents thread teardown races
    "--single-process",              # run Chromium entirely in one process (safe for small apps)
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--ignore-gpu-blacklist",
    "--enable-webgl",
    "--use-gl=angle",              # use ANGLE on Windows (Direct3D)
]

os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = " ".join(flags)
os.environ["QT_QUICK_BACKEND"] = "software"
os.environ["QT_OPENGL"] = "software"
os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false;qt.webenginecontext.*=false"

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    import matplotlib.ticker as ticker
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    WEBENGINE_AVAILABLE = True
except Exception:
    QWebEngineView = None
    WEBENGINE_AVAILABLE = False

try:
    from PyQt6.QtWebEngineCore import QWebEngineProfile
except Exception:
    QWebEngineProfile = None

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal, QEventLoop, QEvent, QUrl
from PyQt6.QtGui import QAction, QIcon, QPalette, QStandardItem, QStandardItemModel, QFontMetrics, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStyledItemDelegate,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from funcs import ( preprocess, create_3_vectors, create_vectors, exposure,
                    score_v_localres, features, getcols, average_score, visualize,
                    score_v_localres_plotly, max_exposure_score, max_m_for_full_matrix, 
                    standard_residues, available_scoring_functions, standard_atoms, all_atoms)

basedir = os.path.dirname(__file__)

def _cleanup_webview(view):
    """Best-effort cleanup of a QWebEngineView / page."""
    if view is None:
        return

    try:
        # stop any loading
        view.stop()
    except Exception:
        pass

    try:
        # try to navigate to about:blank to tear down renderer work
        view.page().setUrl(QUrl("about:blank"))
    except Exception:
        pass

    try:
        # delete page first
        page = view.page()
        if page is not None:
            page.deleteLater()
    except Exception:
        pass

    try:
        view.deleteLater()
    except Exception:
        pass

    # clear profile cache (best-effort)
    try:
        if QWebEngineProfile is not None:
            QWebEngineProfile.defaultProfile().clearHttpCache()
    except Exception:
        pass

    # give Qt a moment to process deletes
    try:
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
            # tiny pause to let background threads stop gracefully
            time.sleep(0.05)
    except Exception:
        pass


class CheckableComboBox(QComboBox):
    '''
    Custom Widget for a combo box where multiple entries can be selected.
    '''
    # Subclass Delegate to increase item height
    class Delegate(QStyledItemDelegate):
        def sizeHint(self, option, index):
            size = super().sizeHint(option, index)
            size.setHeight(20)
            return size

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Make the combo editable to set a custom text, but readonly
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        # Make the lineedit the same color as QPushButton
        palette = QApplication.palette()
        palette.setBrush(QPalette.ColorRole.Base, palette.button())
        self.lineEdit().setPalette(palette)

        # Use custom delegate
        self.setItemDelegate(CheckableComboBox.Delegate())

        # Update the text when an item is toggled
        self.model().dataChanged.connect(self.updateText)

        # Hide and show popup when clicking the line edit
        self.lineEdit().installEventFilter(self)
        self.closeOnLineEditClick = False

        # Prevent popup from closing when clicking on an item
        self.view().viewport().installEventFilter(self)

    def resizeEvent(self, event):
        # Recompute text to elide as needed
        self.updateText()
        super().resizeEvent(event)

    def eventFilter(self, object, event):
        if object == self.lineEdit():
            if event.type() == QEvent.Type.MouseButtonRelease:
                if self.closeOnLineEditClick:
                    self.hidePopup()
                else:
                    self.showPopup()
                return True
            return False

        if object == self.view().viewport():
            if event.type() == QEvent.Type.MouseButtonRelease:
                index = self.view().indexAt(event.pos())
                item = self.model().item(index.row())

                if item.checkState() == Qt.CheckState.Checked:
                    item.setCheckState(Qt.CheckState.Unchecked)
                else:
                    item.setCheckState(Qt.CheckState.Checked)
                return True
        return False

    def showPopup(self):
        super().showPopup()
        # When the popup is displayed, a click on the lineedit should close it
        self.closeOnLineEditClick = True

    def hidePopup(self):
        super().hidePopup()
        # Used to prevent immediate reopening when clicking on the lineEdit
        self.startTimer(100)
        # Refresh the display text when closing
        self.updateText()

    def timerEvent(self, event):
        # After timeout, kill timer, and reenable click on line edit
        self.killTimer(event.timerId())
        self.closeOnLineEditClick = False

    def updateText(self):
        texts = []
        for i in range(self.model().rowCount()):
            if self.model().item(i).checkState() == Qt.CheckState.Checked:
                texts.append(self.model().item(i).text())
        text = ', '.join(texts)

        # Compute elided text (with '...')
        metrics = QFontMetrics(self.lineEdit().font())
        elidedText = metrics.elidedText(text, Qt.TextElideMode.ElideRight, self.lineEdit().width())
        self.lineEdit().setText(elidedText)

    def addItem(self, text, data=None):
        item = QStandardItem()
        item.setText(text)
        if data is None:
            item.setData(text)
        else:
            item.setData(data)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
        item.setData(Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
        self.model().appendRow(item)

    def addItems(self, texts, datalist=None):
        for i, text in enumerate(texts):
            try:
                data = datalist[i]
            except (TypeError, IndexError):
                data = None
            self.addItem(text, data)

    def currentData(self):
        # Return the list of selected items data
        res = []
        for i in range(self.model().rowCount()):
            if self.model().item(i).checkState() == Qt.CheckState.Checked:
                res.append(self.model().item(i).data())
        return res


class CheckableAddComboBox(QComboBox):
    ADD_ROW_ROLE = Qt.ItemDataRole.UserRole + 100   # a role reserved for the Add... flag
    USER_DATA_ROLE = Qt.ItemDataRole.UserRole

    class Delegate(QStyledItemDelegate):
        def sizeHint(self, option, index):
            size = super().sizeHint(option, index)
            size.setHeight(20)
            return size

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setModel(QStandardItemModel(self))
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setStyleSheet("color: black; background: white;")  # force visible text

        self.setItemDelegate(CheckableAddComboBox.Delegate())

        self._add_row_text = "➕ Add..."
        self._has_add_row = False

        self.model().dataChanged.connect(self.updateText)
        self.model().rowsInserted.connect(self.updateText)
        self.model().rowsRemoved.connect(self.updateText)

        self.lineEdit().installEventFilter(self)
        self.view().viewport().installEventFilter(self)
        self.closeOnLineEditClick = False

        self.updateText()

    # ---------- Add-row management ----------
    def ensureAddRow(self):
        if self._has_add_row:
            return
        add_item = QStandardItem(self._add_row_text)
        add_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        # mark *only* the add-row with this special role
        add_item.setData(True, self.ADD_ROW_ROLE)
        self.model().appendRow(add_item)
        self._has_add_row = True

    def removeAddRow(self):
        if not self._has_add_row:
            return
        last = self.model().rowCount() - 1
        if last >= 0:
            self.model().removeRow(last)
        self._has_add_row = False

    # ---------- addItem (explicit roles) ----------
    def addItem(self, text, data=None):
        add_present = self._has_add_row
        if add_present:
            self.removeAddRow()

        item = QStandardItem()
        item.setText(text)  # display text
        # store the user data explicitly in USER_DATA_ROLE
        item.setData(text if data is None else data, self.USER_DATA_ROLE)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
        item.setData(Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
        # Ensure add-row role is not set on normal items:
        # (we don't call item.setData(..., ADD_ROW_ROLE) here)
        self.model().appendRow(item)

        if add_present:
            self.ensureAddRow()

    def addItems(self, texts, datalist=None):
        for i, text in enumerate(texts):
            try:
                data = datalist[i]
            except (TypeError, IndexError):
                data = None
            self.addItem(text, data)

    # ---------- eventFilter ----------
    def eventFilter(self, obj, event):
        if obj == self.lineEdit():
            if event.type() == QEvent.Type.MouseButtonRelease:
                if self.closeOnLineEditClick:
                    self.hidePopup()
                else:
                    self.showPopup()
                return True
            return False

        if obj == self.view().viewport():
            if event.type() == QEvent.Type.MouseButtonRelease:
                index = self.view().indexAt(event.pos())
                if not index.isValid():
                    return False

                row = index.row()
                item = self.model().item(row)
                if item is None:
                    return False

                last_row = self.model().rowCount() - 1
                is_add_row = self._has_add_row and (row == last_row)

                if is_add_row:
                    text, ok = QInputDialog.getText(self, "Add option", "New option:")
                    if ok and text:
                        self.removeAddRow()
                        self.addItem(text)
                        new_index = self.model().rowCount() - 1
                        if new_index >= 0:
                            self.model().item(new_index).setCheckState(Qt.CheckState.Checked)
                        self.ensureAddRow()
                        self.updateText()
                    return True

                if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                    item.setCheckState(
                        Qt.CheckState.Unchecked
                        if item.checkState() == Qt.CheckState.Checked
                        else Qt.CheckState.Checked
                    )
                    self.updateText()
                    return True

                return False

            return False

        return super().eventFilter(obj, event)

    def showPopup(self):
        self.ensureAddRow()
        super().showPopup()
        self.closeOnLineEditClick = True
        self.updateText()

    def hidePopup(self):
        super().hidePopup()
        self.startTimer(100)
        self.updateText()

    def timerEvent(self, event):
        self.killTimer(event.timerId())
        self.closeOnLineEditClick = False

    # ---------- updateText ----------
    def updateText(self, *args, **kwargs):
        selected_texts = []
        row_count = self.model().rowCount()
        for i in range(row_count):
            item = self.model().item(i)
            if item is None:
                continue

            # check *only* our explicit add-row role
            is_add_row = bool(item.data(self.ADD_ROW_ROLE))
            if is_add_row:
                continue

            state = item.checkState()
            if state == Qt.CheckState.Checked:
                selected_texts.append(item.text())

        full_text = ', '.join(selected_texts)

        self.lineEdit().setToolTip(full_text)
        if self.view().isVisible():
            metrics = QFontMetrics(self.lineEdit().font())
            elided = metrics.elidedText(full_text, Qt.TextElideMode.ElideRight, self.lineEdit().width())
            self.lineEdit().setText(elided)
        else:
            self.lineEdit().setText(full_text)

    def currentData(self):
        res = []
        for i in range(self.model().rowCount()):
            item = self.model().item(i)
            if item is None:
                continue
            if bool(item.data(Qt.ItemDataRole.UserRole + 1)):
                continue
            if item.checkState() == Qt.CheckState.Checked:
                res.append(item.text())
        return res
    

class AddableComboBox(QComboBox):
    """QComboBox that looks & behaves like normal single-selection combo
    but adds a permanent 'Add...' row in the popup to add new items."""
    ADD_ROW_ROLE = Qt.ItemDataRole.UserRole + 100

    class Delegate(QStyledItemDelegate):
        def sizeHint(self, option, index):
            size = super().sizeHint(option, index)
            size.setHeight(20)
            return size

    def __init__(self, parent=None):
        super().__init__(parent)

        # Use explicit QStandardItemModel so we can append a custom item
        self.setModel(QStandardItemModel(self))
        self.setItemDelegate(AddableComboBox.Delegate())

        self._add_row_text = "➕ Add..."
        self._has_add_row = False

        # install event filter on popup viewport to intercept clicks
        self.view().viewport().installEventFilter(self)

    # ---------- add-row management ----------
    def ensureAddRow(self):
        """Append the Add... row if not already present."""
        if self._has_add_row:
            return
        add_item = QStandardItem(self._add_row_text)
        add_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        add_item.setData(True, self.ADD_ROW_ROLE)
        self.model().appendRow(add_item)
        self._has_add_row = True

    def removeAddRow(self):
        """Remove the Add... row if present."""
        if not self._has_add_row:
            return
        last = self.model().rowCount() - 1
        if last >= 0:
            self.model().removeRow(last)
        self._has_add_row = False

    # ---------- override addItem/addItems to keep Add row last ----------
    def addItem(self, text, userData=None):
        """Append a normal item, keeping Add... at the end."""
        had_add = self._has_add_row
        if had_add:
            self.removeAddRow()

        # Use QStandardItem so it's part of the model
        item = QStandardItem(text)
        # Make it selectable by default (QComboBox reads display text)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        self.model().appendRow(item)

        if had_add:
            self.ensureAddRow()

    def addItems(self, texts):
        for t in texts:
            self.addItem(t)

    # ---------- event filter: handle clicks on Add... row ----------
    def eventFilter(self, obj, event):
        # Only care about clicks inside the popup viewport
        if obj == self.view().viewport():
            if event.type() == QEvent.Type.MouseButtonRelease:
                index = self.view().indexAt(event.pos())
                if not index.isValid():
                    return False

                row = index.row()
                item = self.model().item(row)
                if item is None:
                    return False

                # Treat only the actual last row as Add... (and only if flagged)
                last_row = self.model().rowCount() - 1
                is_add_row = self._has_add_row and (row == last_row) and bool(item.data(self.ADD_ROW_ROLE))

                if is_add_row:
                    # consume the click and open Add dialog
                    text, ok = QInputDialog.getText(self, "Add item", "New item:")
                    if ok and text:
                        # insert new item before add-row
                        self.removeAddRow()
                        self.addItem(text)
                        # set current selection to the newly added item (last before add-row)
                        new_index = self.model().rowCount() - 1
                        if new_index >= 0:
                            self.setCurrentIndex(new_index)
                        self.ensureAddRow()
                    return True

                # not add-row: return False so Qt will handle normal selection
                return False

            return False

        # fallback for other objects
        return super().eventFilter(obj, event)

    # ---------- show popup ensure Add row ----------
    def showPopup(self):
        # make sure the Add row is present every time popup is shown
        self.ensureAddRow()
        super().showPopup()

    # Optionally expose current items as list
    def items(self):
        out = []
        for i in range(self.model().rowCount()):
            item = self.model().item(i)
            if item is None:
                continue
            if bool(item.data(self.ADD_ROW_ROLE)):
                continue
            out.append(item.text())
        return out


class MplCanvas(FigureCanvas):
    '''Simple Matplotlib FigureCanvas to hold one Axes.'''
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        # let Qt resize the canvas nicely
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()


class MatplotlibWidget(QWidget):
    '''
    Custom Widget for interactive matplotlib plots.
    '''
    def __init__(self, parent=None, initial_image=None):
        super().__init__(parent)
        self.setWindowTitle('Solvent Exposure Calculator')
        self.canvas = MplCanvas(self)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        if initial_image is None:
            initial_image = [[5.96254694e-01, 4.07681084e+00, 5.81393557e+00, 6.37337090e+00, 6.97246239e+00, 7.55877251e+00, 8.14584260e+00, 8.70770365e+00, 1.04676556e+01, 1.45273727e+01, 1.97687311e+01, 2.03432682e+01, 2.09293333e+01, 2.15253676e+01, 5.80172979e-01, 4.07076499e+00, 5.81957257e+00, 1.04847914e+01, 1.45310235e+01, 1.85978996e+01, 1.91832501e+01, 2.15167931e+01, 2.20982095e+01, 5.80238103e-01, 4.06993127e+00, 5.81351430e+00, 1.04680354e+01, 1.45359415e+01, 1.85984438e+01, 2.20905906e+01, 2.26805850e+01, 5.70678639e-01, 4.07527341e+00, 5.83303445e+00, 1.04534222e+01, 1.45208244e+01, 1.86193254e+01, 2.26765641e+01, 5.78263938e-01, 1.16154060e+00, 1.72810016e+00, 2.33523034e+00, 2.89196858e+00, 3.48416418e+00, 4.05792252e+00, 5.83117744e+00, 6.38673299e+00, 6.97187712e+00, 7.54760032e+00, 8.15496711e+00, 8.73808739e+00, 1.04528114e+01, 1.45423796e+01, 1.80054507e+01, 1.86056372e+01, 2.26732865e+01, 5.87125668e-01, 4.06209602e+00, 5.81302133e+00, 1.04602964e+01, 1.45341058e+01, 1.86197518e+01, 2.26821304e+01, 5.99551577e-01, 4.08660491e+00, 5.82505812e+00, 1.04693737e+01, 1.45260620e+01, 1.85860154e+01, 2.20967145e+01, 5.77011881e-01, 4.06241088e+00, 5.80097198e+00, 1.04678560e+01, 1.45427725e+01, 1.85894946e+01, 1.91965206e+01, 2.15076761e+01, 2.20967381e+01, 5.90398520e-01, 4.06416451e+00, 5.82387829e+00, 6.41501907e+00, 6.96543687e+00, 7.56718083e+00, 8.12951134e+00, 8.73874034e+00, 1.04706577e+01, 1.10432308e+01, 1.16205835e+01, 1.22127169e+01, 1.27928050e+01, 1.45351232e+01, 1.51177123e+01, 1.57113429e+01, 1.62770287e+01, 1.68556621e+01, 1.97684793e+01, 2.03543266e+01, 2.09309577e+01, 2.15164564e+01,-1.00742041e-02, 5.78532747e-01, 2.91623156e+00, 5.82090449e+00, 8.14526052e+00, 8.71423458e+00, 9.27625726e+00, 9.88910877e+00, 1.21864962e+01, 1.27846172e+01, 1.33755849e+01, 1.39486357e+01, 1.45525620e+01, 1.51152291e+01, 1.74302199e+01, 2.15277799e+01, 2.20924561e+01, 2.26676117e+01, 2.32499354e+01, 2.38135969e+01, 5.88882615e-01, 2.91057873e+00, 3.48957769e+00, 5.81406085e+00, 6.96808008e+00, 7.56381116e+00, 9.88789902e+00, 1.04615215e+01, 1.22027408e+01, 1.51228671e+01, 1.56963707e+01, 1.74538158e+01, 2.15409496e+01, 2.44365313e+01, 5.78699241e-01, 2.32870793e+00, 2.89692332e+00, 3.50035867e+00, 5.23518255e+00, 5.81933923e+00, 6.98421693e+00, 1.04667339e+01, 1.10497314e+01, 1.22012278e+01, 1.56929122e+01, 1.74391110e+01, 2.15128055e+01, 2.44305668e+01, 2.50040426e+01, 5.75863761e-01, 2.32852158e+00, 3.48798186e+00, 5.22298712e+00, 6.97112787e+00, 1.10429895e+01, 1.21978595e+01, 1.51236639e+01, 1.56895376e+01, 1.74429029e+01, 2.15061073e+01, 2.50005677e+01, 5.83782172e-01, 1.15965315e+00, 2.32462759e+00, 3.48930824e+00, 4.04819444e+00, 5.24007189e+00, 6.40214874e+00, 6.97544752e+00, 1.10422687e+01, 1.22062432e+01, 1.27966220e+01, 1.33575114e+01, 1.39503360e+01, 1.45325285e+01, 1.51299206e+01, 1.74386255e+01, 2.15230918e+01, 2.49919462e+01, 1.17522516e+00, 2.31499717e+00, 4.07821476e+00, 5.23732380e+00, 6.96274490e+00, 1.10465791e+01, 1.22153678e+01, 1.51112733e+01, 1.56984058e+01, 1.74478217e+01, 2.14990555e+01, 2.49976384e+01, 1.16515567e+00, 1.75464083e+00, 4.07066864e+00, 4.67080924e+00, 5.23179231e+00, 6.99319159e+00, 1.04564233e+01, 1.22279745e+01, 1.56991712e+01, 1.74447446e+01, 2.15112740e+01, 2.50028732e+01, 1.15949497e+00, 1.74674112e+00, 4.07759650e+00, 4.65360638e+00, 6.96662454e+00, 7.53631861e+00, 9.87669501e+00, 1.04745337e+01, 1.21970568e+01, 1.56975777e+01, 1.74326713e+01, 2.15305694e+01, 2.44038165e+01, 1.16709334e+00, 1.77184264e+00, 4.63527800e+00, 8.13595770e+00, 8.72082896e+00, 9.30675522e+00, 9.87241616e+00, 1.22187599e+01, 1.57089468e+01, 1.74544736e+01, 1.80216100e+01, 1.86104576e+01, 1.91876478e+01, 1.97733101e+01, 2.15199217e+01, 2.20813260e+01, 2.26771563e+01, 2.32416750e+01, 2.38379756e+01], 
                             [0.50050222, 0.50008824, 0.50002468, 0.50020342, 0.49998496, 0.50001603, 0.50003378, 0.49973335, 0.50016288, 0.50027661, 0.49955886, 0.4999228, 0.50033779, 0.49994516, 0.48684348, 0.48611906, 0.4869761 , 0.48638985, 0.4863771 , 0.48629484, 0.48650368, 0.48606582, 0.48594095, 0.47325186, 0.47266795, 0.47277094, 0.47195167, 0.47284082, 0.47275414, 0.47276174, 0.47243042, 0.45930718, 0.45908547, 0.45889352, 0.45900042, 0.45904937, 0.45850516, 0.45866929, 0.44517504, 0.44516499, 0.44544861, 0.44624171, 0.44591431, 0.4456386 , 0.44526414, 0.44526866, 0.44533243, 0.44621404, 0.44527029, 0.44586661, 0.44521084, 0.44560427, 0.44502731, 0.44554342, 0.44602731, 0.44554905, 0.43186375, 0.43185204, 0.43209572, 0.4308597, 0.43217319, 0.43173451, 0.4313419 , 0.41800185, 0.41815116, 0.41818782, 0.41821606, 0.41761548, 0.41772398, 0.41830504, 0.40475771, 0.40442067, 0.40495383, 0.40479637, 0.40433334, 0.40429696, 0.40487882, 0.40377629, 0.40509251, 0.39078735, 0.39077427, 0.39150869, 0.39098827, 0.39081699, 0.39103901, 0.39114837, 0.39079314, 0.39104908, 0.39116737, 0.39123088, 0.39067675, 0.39129581, 0.3910187 , 0.39081521, 0.39088368, 0.39070704, 0.39087896, 0.39054643, 0.39125805, 0.39164516, 0.39124206, 0.30899095, 0.30924036, 0.30934737, 0.30936133, 0.30877282, 0.3093653 , 0.30911405, 0.30926357, 0.30906188, 0.30946723, 0.30875048, 0.3089421 , 0.30929659, 0.30970443, 0.30916906, 0.30947228, 0.3092335 , 0.30889382, 0.30867412, 0.3088584 , 0.29563449, 0.2952924 , 0.29571118, 0.29575758, 0.29530579, 0.29614876, 0.29536861, 0.29553911, 0.29527109, 0.29563072, 0.29571621, 0.2951226 , 0.29545702, 0.29530559, 0.28239096, 0.28141879, 0.28149263, 0.28203036, 0.28154934, 0.28178658, 0.28195189, 0.28161289, 0.28188398, 0.2817703 , 0.28144152, 0.28188272, 0.28187005, 0.28193321, 0.2821013, 0.26813014, 0.26867725, 0.26735952, 0.26868006, 0.26792643, 0.2685667, 0.26832514, 0.26827841, 0.26804453, 0.26878496, 0.26865959, 0.26795661, 0.25480848, 0.25401354, 0.25443246, 0.25479527, 0.25428121, 0.25463727, 0.25422197, 0.25471256, 0.25425323, 0.25375204, 0.25408643, 0.25420654, 0.25467597, 0.25481153, 0.25474452, 0.25460552, 0.25448885, 0.2543339, 0.24098642, 0.24042494, 0.24087567, 0.24098655, 0.24080232, 0.2412132, 0.24044209, 0.24083115, 0.24115701, 0.24072121, 0.24093191, 0.24040232, 0.22742359, 0.22741627, 0.22676936, 0.22787531, 0.22733598, 0.22718892, 0.22678687, 0.22671446, 0.22695256, 0.22765442, 0.2274607 , 0.22700747, 0.21371944, 0.21364745, 0.21384062, 0.21336159, 0.2133576 , 0.21357356, 0.21298157, 0.2135119 , 0.21387164, 0.21340503, 0.21355778, 0.21415836, 0.21329659, 0.19968525, 0.20019266, 0.19976646, 0.19995692, 0.20023523, 0.20024174, 0.20035051, 0.19980486, 0.2000605 , 0.19993717, 0.20048241, 0.20049686, 0.2002386 , 0.19954422, 0.20012563, 0.1997846 , 0.20048284, 0.19984043, 0.19973371]]

        self.canvas.ax.scatter(*initial_image)
        self._last_recip_update = 0.0
        self._mpl_cursor = None
        self._connect_tick_callbacks_once()
        self.canvas.figure.tight_layout()
        self.canvas.draw_idle()

    def reciprocal_ticks(self, mn, mx, n=4, intervals=[1, 2, 5, 10, 20, 50, 100]):
        '''
        Same premise as funcs.reciprocal_ticks, but designed for GUI use.
        '''
        ticks = []
        if mn == mx:
            return np.array([])
        low = min(mn, mx)
        high = max(mn, mx)
        if high <= 0:
            return np.array([])
        if low <= 0:
            low = 1e-12
        for i in intervals:
            if i/low - i/high < n:
                continue
            else:
                tick = np.ceil(i/low) / i
                while tick > 1/high:
                    ticks.append(tick)
                    tick -= 1.0 / i
                break
        return np.array(sorted(ticks))

    def _update_reciprocal_y_ticks(self, ax=None):
        '''Compute labels and set reciprocal-style y-ticks. Debounced to avoid frequent runs.'''
        # debounce: allow at most ~6-7 updates per second
        now = time.time()
        if now - getattr(self, '_last_recip_update', 0.0) < 0.15:
            return
        self._last_recip_update = now

        if ax is None:
            ax = self.canvas.ax
        ymin, ymax = ax.get_ylim()
        if ymax <= 0:
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{y:g}'))
            self.canvas.draw_idle()
            return

        ticks = self.reciprocal_ticks(ymin, ymax)
        if ticks.size == 0:
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'1/{1/y:.3g}' if y != 0 else '0'))
        else:
            ax.set_yticks(1/ticks)
            labels = []
            for val in ticks:
                if val == 0:
                    labels.append('0')
                else:
                    denom = val
                    if abs(denom - round(denom)) < 1e-8:
                        labels.append(f'1/{int(round(denom))}')
                    else:
                        labels.append(f'1/{denom:g}')
            ax.set_yticklabels(labels)

        # draw once after updates (debounced)
        self.canvas.draw_idle()

    def _connect_tick_callbacks_once(self):
        ax = self.canvas.ax
        if getattr(self, '_recip_callbacks_connected', False):
            return
        # connect to ylim_changed using a lambda that calls debounced updater
        ax.callbacks.connect('ylim_changed', lambda a: self._update_reciprocal_y_ticks(a))
        # also connect to draw_event as a fallback
        self.canvas.mpl_connect('draw_event', lambda event: self._update_reciprocal_y_ticks())
        self._recip_callbacks_connected = True

    def plot_score_vs_resolution(self, x, y, names=None, xlabel=None, ylabel=None):
        ax = self.canvas.ax
        ax.clear()

        sc = ax.scatter(x, y, s=10)

        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        # Clean up previous cursor if present
        try:
            if self._mpl_cursor is not None:
                self._mpl_cursor.disconnect()
        except Exception:
            pass

        # Reset active annotation holder
        self._active_annotation = None

        # Create an mplcursors cursor for this scatter collection
        self._mpl_cursor = mplcursors.cursor(sc, hover=True)

        def _on_add(sel):
            i = sel.index
            if names is not None:
                try:
                    ch, resatom = names[i].split(':',1)
                    label = f"Chain {ch[1:]}\nResidue {resatom.split('@',1)[0]}\nAtom {resatom.split('@',1)[1]}"
                except Exception:
                    label = str(i)
            else:
                label = str(i)

            # set text and style, and ensure visible
            sel.annotation.set_text(label)
            sel.annotation.get_bbox_patch().set_alpha(0.85)
            sel.annotation.set_visible(True)

            # hide any previously shown annotation (so old labels don't linger)
            if getattr(self, '_active_annotation', None) is not None and self._active_annotation is not sel.annotation:
                try:
                    self._active_annotation.set_visible(False)
                except Exception:
                    pass

            self._active_annotation = sel.annotation
            self.canvas.draw_idle()

        def _on_remove(sel):
            try:
                sel.annotation.set_visible(False)
            except Exception:
                pass

            if getattr(self, '_active_annotation', None) is sel.annotation:
                self._active_annotation = None

            self.canvas.draw_idle()

        _last_motion = {'t': 0.0}
        def _on_motion(event):
            # Only do work if an annotation is currently visible (common case: most motion is ignored)
            if getattr(self, '_active_annotation', None) is None:
                return

            # Throttle checks to reduce CPU (max ~10 fps here)
            now = time.time()
            if now - _last_motion['t'] < 0.1: 
                return
            _last_motion['t'] = now

            if event.inaxes != self.canvas.ax:
                # Moved outside axes — hide active annotation
                try:
                    if self._active_annotation is not None:
                        self._active_annotation.set_visible(False)
                        self._active_annotation = None
                        self.canvas.draw_idle()
                except Exception:
                    pass
                return

            # Check whether the mouse is over any scatter point
            try:
                contains, info = sc.contains(event)
            except Exception:
                contains = False

            if not contains:
                try:
                    if self._active_annotation is not None:
                        self._active_annotation.set_visible(False)
                        self._active_annotation = None
                        self.canvas.draw_idle()
                except Exception:
                    pass

        self._motion_cid = self.canvas.mpl_connect('motion_notify_event', _on_motion)
        self._mpl_cursor.connect('add', _on_add)
        self._mpl_cursor.connect('remove', _on_remove)
        self._update_reciprocal_y_ticks()
        self._connect_tick_callbacks_once()
        ax.figure.tight_layout()
        self._active_annotation = None
        self.canvas.draw_idle()


class ScriptWorker(QObject):
    '''
    Class which contains scripts that are run for calculations and the signalling to enable communication.
    '''
    started = pyqtSignal()
    finished = pyqtSignal(object)   # can emit results
    error = pyqtSignal(str)
    ask = pyqtSignal(str)
    answer = pyqtSignal(bool)
    progress = pyqtSignal(str)

    def __init__(self, settings):
        super().__init__()
        self.settings = settings  # dict of settings

    def run_simple(self):
        '''
        Script which runs simple solvent exposure calculation for an inputted pdb or mmcif file.
        '''
        self.started.emit()
        try:
            settings = self.settings
            
            pdb_path = settings.get('pdb_path')
            pre_path = settings.get('folder_pre_path')
            out_path = settings.get('folder_out_path')
            average = settings.get('average')
            backbone = settings.get('backbone')
            weight = settings.get('weight_by_atomic_mass')
            pickle_pre = settings.get('pickle_preprocessed')

            pre_out = preprocess(pdb_path=pdb_path, pre_path=pre_path, yn=self.yes_no, pickle_out=pickle_pre)
            self.progress.emit('Preprocessing complete')
            result = exposure(pdb_path=pre_out, out_path=out_path, yn=self.yes_no, weight_by_amu=weight, progress_callback=self.progress.emit)
            if average:
                self.progress.emit('Main calculation(s) complete.')
                tempresult = []
                for i in result:
                    tempresult += average_score(i[0], backbone=backbone)
                result += tempresult

            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

    def run_adduct_pre(self):
        '''
        Script which runs preprocessing for solvent exposure calculation, quantifying the contributions from a subset of atoms.
        '''
        self.started.emit()
        try:
            settings = self.settings
            pdb_path = settings.get('pdb_path')
            pre_path = settings.get('folder_pre_path')
            feature = settings.get('feature')
            pickle_pre = settings.get('pickle_preprocessed')

            pre_out = preprocess(pdb_path=pdb_path, pre_path=pre_path, yn=self.yes_no, pickle_out=pickle_pre)
            self.progress.emit('Preprocessing complete')
            if pre_out.rsplit('.',1)[1] == 'cif':
                if feature == 'chain_id':
                    feature = '_atom_site.label_asym_id'
                elif feature == 'residue_name':
                    feature = '_atom_site.label_comp_id'
            features_out = features(pdb_path=pre_out, feature=feature)
            result = pre_out, features_out
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

    def run_adduct_out(self):
        '''
        Script which runs the solvent exposure calculation, quantifying the contributions from a subset of atoms.
        '''
        self.started.emit()
        try:
            settings = self.settings
            pdb_path = settings.get('pre_out_path')
            out_path = settings.get('folder_out_path')
            feature = settings.get('feature')
            combo = settings.get('combo')
            average = settings.get('average')
            backbone = settings.get('backbone')
            weight = settings.get('weight_by_atomic_mass')

            if pdb_path.rsplit('.',1)[1] == 'cif':
                if feature == 'chain_id':
                    feature = '_atom_site.label_asym_id'
                elif feature == 'residue_name':
                    feature = '_atom_site.label_comp_id'

            assignment = create_3_vectors(pdb_path=pdb_path, chain1=combo, feature=feature)
            result = exposure(pdb_path=pdb_path, out_path=out_path, yn=self.yes_no, assignment=assignment, weight_by_amu=weight, progress_callback=self.progress.emit)
            if average:
                self.progress.emit('Main calculation(s) complete.')
                tempresult = []
                for i in result:
                    tempresult += average_score(i[0], backbone=backbone)
                result += tempresult

            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

    def run_plot(self):
        '''
        Returns the entries necessary for the GUI to plot local resolution versus solvent exposure score.
        '''
        self.started.emit()
        try:
            settings = self.settings
            pdb_path = settings.get('pdb_path')
            defattr_path = settings.get('defattr_path')
            only_chain = settings.get('only_chain')
            only_backbone = settings.get('only_backbone')

            result = score_v_localres(pdb_path=pdb_path, defattr_path=defattr_path, only_chain=only_chain, called_by_GUI=True, backboneonly=only_backbone, inverse=True)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

    def run_manual_preprocessing(self):
        '''
        Script which runs custom preprocessing.
        '''
        self.started.emit()
        try:
            settings = self.settings
            pdb_path = settings.get('preprocess_file_path')
            pre_path = settings.get('preprocess_folder_path')
            # include = settings.get('preprocess_include_selected')
            redefine_chains = settings.get('preprocess_redefine_chains')
            pickle_pre = settings.get('pickle_preprocessed')

            result = preprocess(pdb_path=pdb_path, pre_path=pre_path, yn=self.yes_no, redefine_chains=redefine_chains, pickle_out=pickle_pre)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

    def run_manual_calculation(self):
        '''
        Script which runs custom solvent exposure calculation
        '''
        self.started.emit()
        try:
            settings = self.settings
            pdb_path = settings.get('preprocessed_path_calculate')
            out_path = settings.get('calculate_folder_path')
            if settings.get('calculate_assignment'):
                assignment = settings.get('assignment_vectors')
            else:
                assignment = None
            average = settings.get('average')
            backbone = settings.get('backbone')
            funcs = settings.get('funcs')
            weight = settings.get('weight_by_atomic_mass')
            
            result = exposure(pdb_path=pdb_path, out_path=out_path, yn=self.yes_no, assignment=assignment, funcs=funcs, weight_by_amu=weight, progress_callback=self.progress.emit)
            if average:
                self.progress.emit('Main calculation(s) complete.')
                tempresult = []
                for i in result:
                    tempresult += average_score(i[0], backbone=backbone)
                result += tempresult
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
        
    def yes_no(self, text):
        '''
        Sets up the communication for a popup yes/no window.
        '''
        loop = QEventLoop()
        response_holder = {'val': None}

        def _on_answer(val):
            response_holder['val'] = val
            loop.quit()

        # connect temporary slot
        self.answer.connect(_on_answer)
        # emit request to main thread
        self.ask.emit(text)
        # block worker thread until main thread answers
        loop.exec()
        # cleanup connection
        try:
            self.answer.disconnect(_on_answer)
        except Exception:
            pass

        return bool(response_holder['val'])


class MainWindow(QMainWindow):
    '''
    Main window of the application, as well as the functions that enable communication between GUI elements and running scripts.
    '''
    def __init__(self):
        '''
        Main Window Layout
        '''
        super().__init__()
        self.setWindowTitle('Solvent Exposure Calculation')
        self.resize(600, 800)

        self.enable_disable = []
        self.all_settings = []
        self.weight_by_atomic_mass = True
        self.pickle_preprocessed = False

        file_menu = self.menuBar().addMenu('&File')
        close_action = QAction('Close', self)
        close_action.setShortcut(QKeySequence(QKeySequence.StandardKey.Close))  
        close_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut) 
        close_action.triggered.connect(self.close)
        file_menu.addAction(close_action)
        self.addAction(close_action)

        settings_menu = self.menuBar().addMenu('&Settings')
        settings_menu.setToolTipsVisible(True)
        self.weight_by_atomic_mass_button = QAction("&Weight by atomic mass", self)
        self.weight_by_atomic_mass_button.setToolTip("Would you like to weight contributions to score by atomic mass? This will change max score.")
        self.weight_by_atomic_mass_button.triggered.connect(self.weight_by_atomic_mass_clicked)
        self.weight_by_atomic_mass_button.setCheckable(True)
        self.weight_by_atomic_mass_button.setChecked(self.weight_by_atomic_mass)
        self.enable_disable.append(self.weight_by_atomic_mass_button)
        settings_menu.addAction(self.weight_by_atomic_mass_button)

        self.pickle_preprocessed_button = QAction('&Compress Preprocessed', self)
        self.pickle_preprocessed_button.setToolTip("Would you like to compress preprocessed files? This will speed up file saving/reading, but makes troubleshooting more difficult.")
        self.pickle_preprocessed_button.triggered.connect(self.pickle_preprocessed_clicked)
        self.pickle_preprocessed_button.setCheckable(True)
        self.pickle_preprocessed_button.setChecked(self.pickle_preprocessed)
        self.enable_disable.append(self.pickle_preprocessed_button)
        settings_menu.addAction(self.pickle_preprocessed_button)

        # Set up Tabs
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.TabPosition.North)
        tabs.setMovable(False)

        ###
        # Simple Tab
        ###
        simple = QWidget()
        simple_form = QVBoxLayout()

        self.current_simple_settings = {
            'pdb_path': os.path.join(basedir, 'pdbs', 'in', '1u7g.pdb'),
            'folder_pre_path': os.path.join(basedir, 'pdbs', 'preprocessed'),
            'folder_out_path': os.path.join(basedir, 'pdbs', 'out'),
            'average': True,
            'backbone': False,
            'weight_by_atomic_mass': self.weight_by_atomic_mass,
            'pickle_preprocessed': self.pickle_preprocessed,
        }
        self.all_settings.append(self.current_simple_settings)

        self.simple_average = True
        self.simple_backbone = False

        # PDB file selection
        file_row = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setText(self.current_simple_settings.get('pdb_path', ''))
        self.file_browse = QPushButton('Browse...')
        self.file_browse.clicked.connect(self._browse_file)
        file_row.addWidget(QLabel('PDB/mmCIF File:'))
        file_row.addWidget(self.file_edit)
        file_row.addWidget(self.file_browse)
        simple_form.addLayout(file_row)

        # Folder (preproccessed) selection
        folder_pre_row = QHBoxLayout()
        self.folder_pre_edit = QLineEdit()
        self.folder_pre_edit.setText(self.current_simple_settings.get('folder_pre_path', ''))
        self.folder_pre_browse = QPushButton('Browse...')
        self.folder_pre_browse.clicked.connect(self._browse_pre_folder)
        folder_pre_row.addWidget(QLabel('Preprocessed Folder:'))
        folder_pre_row.addWidget(self.folder_pre_edit)
        folder_pre_row.addWidget(self.folder_pre_browse)
        simple_form.addLayout(folder_pre_row)

        # Folder (out) selection
        folder_out_row = QHBoxLayout()
        self.folder_out_edit = QLineEdit()
        self.folder_out_edit.setText(self.current_simple_settings.get('folder_out_path', ''))
        self.folder_out_browse = QPushButton('Browse...')
        self.folder_out_browse.clicked.connect(self._browse_out_folder)
        folder_out_row.addWidget(QLabel('Output Folder:'))
        folder_out_row.addWidget(self.folder_out_edit)
        folder_out_row.addWidget(self.folder_out_browse)
        simple_form.addLayout(folder_out_row)

        # Average checkbox
        simple_average_row = QHBoxLayout()

        self.simple_average_checkbox = QCheckBox('Average output per residue?')
        self.simple_average_checkbox.setChecked(self.simple_average)
        self.simple_average_checkbox.stateChanged.connect(self._on_simple_average_toggled)
        simple_average_row.addWidget(self.simple_average_checkbox)
        self.simple_backbone_checkbox = QCheckBox('Also average only by backbone?')
        self.simple_backbone_checkbox.setChecked(self.simple_backbone)
        self.simple_backbone_checkbox.stateChanged.connect(self._on_simple_backbone_toggled)        
        simple_average_row.addWidget(self.simple_backbone_checkbox)
        simple_form.addLayout(simple_average_row)

        # Output text box
        self.simple_output = QTextEdit()
        self.simple_output.setReadOnly(True)
        self.simple_output.setPlaceholderText('Results will appear here...')
        simple_form.addWidget(self.simple_output)

        # Bottom: Run Button
        self.run_simple = QPushButton('Calculate')
        self.run_simple.clicked.connect(self.on_run_simple_clicked)
        self.enable_disable.append(self.run_simple)
        simple_form.addWidget(self.run_simple)

        # Add to tab
        simple.setLayout(simple_form)
        tabs.addTab(simple, 'Simple')


        ###
        # Adduct Tab
        ###
        self.adduct_average = True
        self.adduct_backbone = False
        adduct = QWidget()
        adduct_form = QVBoxLayout()

        self.current_adduct_settings = {
            'pdb_path': os.path.join(basedir, 'pdbs', 'in', '1u7g.pdb'),
            'folder_pre_path': os.path.join(basedir, 'pdbs', 'preprocessed'),
            'feature': 'chain_id',
            'pre_out_path': '',
            'folder_out_path': os.path.join(basedir, 'pdbs', 'out'),
            'combo': '',
            'average': True,
            'backbone': False,
            'weight_by_atomic_mass': self.weight_by_atomic_mass,
            'pickle_preprocessed': self.pickle_preprocessed,
        }
        self.all_settings.append(self.current_adduct_settings)

        # PDB File selection
        adduct_file_row = QHBoxLayout()
        self.adduct_file_edit = QLineEdit()
        self.adduct_file_edit.setText(self.current_adduct_settings.get('pdb_path', ''))
        self.adduct_file_browse = QPushButton('Browse...')
        self.adduct_file_browse.clicked.connect(self._browse_adduct_file)
        adduct_file_row.addWidget(QLabel('PDB/mmCIF File:'))
        adduct_file_row.addWidget(self.adduct_file_edit)
        adduct_file_row.addWidget(self.adduct_file_browse)
        adduct_form.addLayout(adduct_file_row)

        # Folder (preprocessed) selection
        adduct_folder_pre_row = QHBoxLayout()
        self.adduct_folder_pre_edit = QLineEdit()
        self.adduct_folder_pre_edit.setText(self.current_adduct_settings.get('folder_pre_path', ''))
        self.adduct_folder_pre_browse = QPushButton('Browse...')
        self.adduct_folder_pre_browse.clicked.connect(self._browse_adduct_pre_folder)
        adduct_folder_pre_row.addWidget(QLabel('Preprocessed Folder:'))
        adduct_folder_pre_row.addWidget(self.adduct_folder_pre_edit)
        adduct_folder_pre_row.addWidget(self.adduct_folder_pre_browse)
        adduct_form.addLayout(adduct_folder_pre_row)

        # Feature selection
        adduct_feature_row = QHBoxLayout()
        self.adduct_feature = QComboBox()
        self.adduct_feature.addItems(['chain_id', 'residue_name'])
        op = self.current_adduct_settings.get('feature', 'chain_id')
        idx = self.adduct_feature.findText(op)
        if idx >= 0:
            self.adduct_feature.setCurrentIndex(idx)
        adduct_feature_row.addWidget(QLabel('Feature:'))
        adduct_feature_row.addWidget(self.adduct_feature)
        adduct_form.addLayout(adduct_feature_row)

        # Preprocess/feature Run Button
        self.run_adduct_pre = QPushButton('Preprocess')
        self.run_adduct_pre.clicked.connect(self.on_run_adduct_pre_clicked)
        self.enable_disable.append(self.run_adduct_pre)
        adduct_form.addWidget(self.run_adduct_pre)

        # Combo selector
        combo_row = QHBoxLayout()
        self.combo = CheckableComboBox()
        combo_row.addWidget(QLabel('Combo:'))
        combo_row.addWidget(self.combo)
        adduct_form.addLayout(combo_row)

        # Folder (out) selection
        adduct_folder_out_row = QHBoxLayout()
        self.adduct_folder_out_edit = QLineEdit()
        self.adduct_folder_out_edit.setText(self.current_adduct_settings.get('folder_out_path', ''))
        self.adduct_folder_out_browse = QPushButton('Browse...')
        self.adduct_folder_out_browse.clicked.connect(self._browse_adduct_out_folder)
        adduct_folder_out_row.addWidget(QLabel('Output Folder:'))
        adduct_folder_out_row.addWidget(self.adduct_folder_out_edit)
        adduct_folder_out_row.addWidget(self.adduct_folder_out_browse)
        adduct_form.addLayout(adduct_folder_out_row)

        # Average checkboxes
        adduct_average_row = QHBoxLayout()
        self.adduct_average_checkbox = QCheckBox('Average output per residue?')
        self.adduct_average_checkbox.setChecked(self.adduct_average)
        self.adduct_average_checkbox.stateChanged.connect(self._on_adduct_average_toggled)
        adduct_average_row.addWidget(self.adduct_average_checkbox)
        self.adduct_backbone_checkbox = QCheckBox('Also average only by backbone?')
        self.adduct_backbone_checkbox.setChecked(self.adduct_backbone)
        self.adduct_backbone_checkbox.stateChanged.connect(self._on_adduct_backbone_toggled)        
        adduct_average_row.addWidget(self.adduct_backbone_checkbox)
        adduct_form.addLayout(adduct_average_row)

        # Bottom: Run Button
        self.run_adduct_out = QPushButton('Calculate')
        self.run_adduct_out.clicked.connect(self.on_run_adduct_out_clicked)
        self.enable_disable.append(self.run_adduct_out)
        adduct_form.addWidget(self.run_adduct_out)

        # Output text box
        self.adduct_output = QTextEdit()
        self.adduct_output.setReadOnly(True)
        self.adduct_output.setPlaceholderText('Results will appear here...')
        adduct_form.addWidget(self.adduct_output)

        adduct.setLayout(adduct_form)
        tabs.addTab(adduct, 'With Adduct')


        ###
        # Visualisation Tab
        ###
        visuals_widget = QWidget()
        visuals_layout = QVBoxLayout()

        # Top controls: PDB path and a render button
        controls_h = QHBoxLayout()
        self.visuals_pdb_edit = QLineEdit()
        self.visuals_pdb_edit.setText(os.path.join(basedir, 'pdbs', 'out', '1u7g_2c50.pdb'))  # default
        self.visuals_browse = QPushButton('Browse...')
        self.visuals_browse.clicked.connect(self._browse_visuals_file)
        controls_h.addWidget(QLabel('PDB/mmCIF:'))
        controls_h.addWidget(self.visuals_pdb_edit)
        controls_h.addWidget(self.visuals_browse)

        # Render / Open controls
        render_h = QHBoxLayout()
        self.visuals_render_btn = QPushButton('Render')
        self.enable_disable.append(self.visuals_render_btn)
        render_h.addWidget(self.visuals_render_btn)

        visuals_layout.addLayout(controls_h)
        visuals_layout.addLayout(render_h)

        # area to show the figure
        if WEBENGINE_AVAILABLE:
            self.visuals_view = QWebEngineView()
            self.visuals_view.setMinimumHeight(480)
            visuals_layout.addWidget(self.visuals_view)
        else:
            # fallback: show a label explaining missing dependency
            self.visuals_info_label = QLabel("Qt WebEngine not available. Use 'Open in Browser' to view the figure externally.")
            self.visuals_info_label.setWordWrap(True)
            visuals_layout.addWidget(self.visuals_info_label)

        visuals_widget.setLayout(visuals_layout)
        tabs.addTab(visuals_widget, 'Protein Visualisation')

        self.visuals_render_btn.clicked.connect(self._render_embed)


        ###
        # Plotting Tab
        ###
        self.only_chain = False
        self.only_backbone = False
        plot = QWidget()
        plot_form = QVBoxLayout()

        self.current_plot_settings = {
            'pdb_path': os.path.join(basedir, 'pdbs', 'out', '3jcz_2c50_26p5.pdb'),
            'defattr_path': os.path.join(basedir, 'pdbs', 'out', 'defattrs', 'gdh_J123.defattr'),
            'only_chain': False,
            'only_backbone': False,
            'weight_by_atomic_mass': self.weight_by_atomic_mass,
            'pickle_preprocessed': self.pickle_preprocessed,
        }

        plot_pdb_row = QHBoxLayout()
        self.plot_pdb_edit = QLineEdit()
        self.plot_pdb_edit.setText(self.current_plot_settings.get('pdb_path', ''))
        self.plot_pdb_browse = QPushButton('Browse...')
        self.plot_pdb_browse.clicked.connect(self._browse_plot_file)
        plot_pdb_row.addWidget(QLabel('PDB/mmCIF File:'))
        plot_pdb_row.addWidget(self.plot_pdb_edit)
        plot_pdb_row.addWidget(self.plot_pdb_browse)
        plot_form.addLayout(plot_pdb_row)

        plot_defattr_row = QHBoxLayout()
        self.plot_defattr_edit = QLineEdit()
        self.plot_defattr_edit.setText(self.current_plot_settings.get('defattr_path', ''))
        self.plot_defattr_browse = QPushButton('Browse...')
        self.plot_defattr_browse.clicked.connect(self._browse_defattr_file)
        plot_defattr_row.addWidget(QLabel('defattr File:'))
        plot_defattr_row.addWidget(self.plot_defattr_edit)
        plot_defattr_row.addWidget(self.plot_defattr_browse)
        plot_form.addLayout(plot_defattr_row)

        # Only chain(s)?
        only_chains_h = QHBoxLayout()
        self.only_chain_checkbox = QCheckBox('Only Chain(s)?')
        self.only_chain_checkbox.setChecked(self.only_chain)
        self.only_chain_checkbox.stateChanged.connect(self._on_only_chains_toggled)
        only_chains_h.addWidget(self.only_chain_checkbox)
        self.only_chain_combo = CheckableComboBox()
        self.only_chain_combo.addItems(features(pdb_path=self.plot_pdb_edit.text(), feature='chain_id'))
        only_chains_h.addWidget(self.only_chain_combo)
        self.only_backbone_checkbox = QCheckBox('Only Backbone Atoms?')
        self.only_backbone_checkbox.setChecked(self.only_backbone)
        self.only_backbone_checkbox.stateChanged.connect(self._on_only_backbones_toggled)
        only_chains_h.addWidget(self.only_backbone_checkbox)

        plot_form.addLayout(only_chains_h)

        # Plot button
        self.run_plot = QPushButton('Plot')
        self.enable_disable.append(self.run_plot)
        plot_form.addWidget(self.run_plot)

        if WEBENGINE_AVAILABLE:
            self.plot_view = QWebEngineView()
            self.plot_view.setMinimumHeight(480)
            plot_form.addWidget(self.plot_view)
            self.run_plot.clicked.connect(self._render_plot_embed)

            plot.setLayout(plot_form)
            tabs.addTab(plot, 'Score vs Resolution')

        elif MATPLOTLIB_AVAILABLE:
            self.sc = MatplotlibWidget(self)
            self.run_plot.clicked.connect(self.on_run_plot_clicked)
            plot_form.addWidget(self.sc)

            # Output text box
            self.plot_output = QTextEdit()
            self.plot_output.setReadOnly(True)
            self.plot_output.setPlaceholderText('Results will appear here...')
            plot_form.addWidget(self.plot_output)

            plot.setLayout(plot_form)
            tabs.addTab(plot, 'Score vs Resolution')
        else:
            tabs.addTab(QLabel('Cannot Plot. QtWebEngine and Matplotlib are not installed'), 'Score vs Resolution')


        ###
        # Manual Tab
        ###
        manual_tot = QWidget()
        manual_form = QVBoxLayout()
        manual = QTabWidget()
        manual.setTabPosition(QTabWidget.TabPosition.West)
        manual.setMovable(False)

        self.current_manual_settings = {
            'preprocess_file_path': os.path.join(basedir, 'pdbs', 'in', '1u7g.pdb'),
            'preprocess_folder_path': os.path.join(basedir, 'pdbs', 'preprocessed'),
            'preprocess_redefine_chains': False,
            'preprocessed_path_assignment': '',
            'single_include': '',
            'single_feature': '',
            'three_include': '',
            'three_feature': '',
            'vector_name': 'name',
            'assignment_vectors': None,
            'preprocessed_path_calculate': '',
            'calculate_folder_path': os.path.join(basedir, 'pdbs', 'out'),
            'function_types': list(available_scoring_functions.keys()),
            'function_selected': 'Close and far cutoff',
            'calculate_assignment': False,
            'average': True,
            'backbone': False,
            'weight_by_atomic_mass': self.weight_by_atomic_mass,
            'pickle_preprocessed': self.pickle_preprocessed,
        }
        self.current_manual_settings['function'] = available_scoring_functions[self.current_manual_settings.get('function_selected')]
        self.all_settings.append(self.current_manual_settings)

        self.preprocess_redefine_chains = False
        self.calculate_assignment = False
        self.calculate_average = True
        self.calculate_backbone = False


        ###
        # Manual Preprocessing
        ###
        manual_preprocess = QWidget()
        manual_preprocess_form = QVBoxLayout()

        # PDB file selection
        manual_preprocess_file_row = QHBoxLayout()
        self.manual_preprocess_file_edit = QLineEdit()
        self.manual_preprocess_file_edit.setText(self.current_manual_settings.get('preprocess_file_path', ''))
        self.manual_preprocess_file_browse = QPushButton('Browse...')
        self.manual_preprocess_file_browse.clicked.connect(self._browse_manual_preprocess_file)
        manual_preprocess_file_row.addWidget(QLabel('PDB/mmCIF File:'))
        manual_preprocess_file_row.addWidget(self.manual_preprocess_file_edit)
        manual_preprocess_file_row.addWidget(self.manual_preprocess_file_browse)
        manual_preprocess_form.addLayout(manual_preprocess_file_row)

        # Folder selection
        manual_preprocess_folder_row = QHBoxLayout()
        self.manual_preprocess_folder_edit = QLineEdit()
        self.manual_preprocess_folder_edit.setText(self.current_manual_settings.get('preprocess_folder_path', ''))
        self.manual_preprocess_folder_browse = QPushButton('Browse...')
        self.manual_preprocess_folder_browse.clicked.connect(self._browse_manual_preprocess_folder)
        manual_preprocess_folder_row.addWidget(QLabel('Preprocessed Folder:'))
        manual_preprocess_folder_row.addWidget(self.manual_preprocess_folder_edit)
        manual_preprocess_folder_row.addWidget(self.manual_preprocess_folder_browse)
        manual_preprocess_form.addLayout(manual_preprocess_folder_row)

        # Relabel chains checkbox
        self.preprocess_redefine_chains_checkbox = QCheckBox('Relabel chains alphabetically?')
        self.preprocess_redefine_chains_checkbox.setChecked(self.preprocess_redefine_chains)
        self.preprocess_redefine_chains_checkbox.stateChanged.connect(self._on_preprocess_redefine_chains_toggled)
        manual_preprocess_form.addWidget(self.preprocess_redefine_chains_checkbox)

        # Preprocess button
        self.run_manual_preprocess = QPushButton('Preprocess')
        self.run_manual_preprocess.clicked.connect(self.on_run_manual_preprocess_clicked)
        self.enable_disable.append(self.run_manual_preprocess)
        manual_preprocess_form.addWidget(self.run_manual_preprocess)

        manual_preprocess.setLayout(manual_preprocess_form)
        manual.addTab(manual_preprocess, 'Preprocess')

        ###
        # Manual Assignment Vector Creation
        ###
        manual_assignment = QWidget()
        manual_assignment_form = QVBoxLayout()

        # Reset Button
        self.manual_assignment_reset = QPushButton('Clear Assignment Vectors')
        self.manual_assignment_reset.clicked.connect(self.on_manual_assignment_reset_clicked)
        self.enable_disable.append(self.manual_assignment_reset)
        manual_assignment_form.addWidget(self.manual_assignment_reset)

        # File selection
        manual_assignment_file_row = QHBoxLayout()
        self.manual_assignment_file_edit = QLineEdit()
        self.manual_assignment_file_edit.setText(self.current_manual_settings.get('preprocessed_path_assignment', ''))
        self.manual_assignment_file_browse = QPushButton('Browse...')
        self.manual_assignment_file_browse.clicked.connect(self._browse_manual_assignment_file)
        manual_assignment_file_row.addWidget(QLabel('PDB/mmCIF File:'))
        manual_assignment_file_row.addWidget(self.manual_assignment_file_edit)
        manual_assignment_file_row.addWidget(self.manual_assignment_file_browse)
        manual_assignment_form.addLayout(manual_assignment_file_row)

        # Vector Creation options
        assignment_creation_tabs = QTabWidget()
        assignment_creation_tabs.setTabPosition(QTabWidget.TabPosition.North)
        assignment_creation_tabs.setMovable(False)

        #Single Vector
        single_vector_tab = QWidget()
        single_vector_layout = QVBoxLayout()

        single_vector_feature_row = QHBoxLayout()
        single_vector_feature_sel_col = QVBoxLayout()
        label = QLabel('Feature')
        label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        single_vector_feature_sel_col.addWidget(label)
        self.manual_single_feature_combo = QComboBox()
        self.manual_single_feature_combo.activated.connect(self._on_single_feature_changed)
        single_vector_feature_sel_col.addWidget(self.manual_single_feature_combo)
        single_vector_feature_row.addLayout(single_vector_feature_sel_col)

        single_vector_include_col = QVBoxLayout()
        label = QLabel('Include')
        label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        single_vector_include_col.addWidget(label)
        self.manual_single_include_combo = CheckableComboBox()
        single_vector_include_col.addWidget(self.manual_single_include_combo)
        single_vector_feature_row.addLayout(single_vector_include_col)
        single_vector_layout.addLayout(single_vector_feature_row)

        single_vector_run_row = QHBoxLayout()
        self.manual_single_assignment_overwrite = QPushButton('Overwrite assignment vector(s)')
        self.manual_single_assignment_overwrite.clicked.connect(lambda _, o = True: self.on_manual_single_assignment_clicked(o))
        self.enable_disable.append(self.manual_single_assignment_overwrite)
        single_vector_run_row.addWidget(self.manual_single_assignment_overwrite)
        self.manual_single_assignment_add = QPushButton('Add assignment vector(s)')
        self.manual_single_assignment_add.clicked.connect(lambda _, o = False: self.on_manual_single_assignment_clicked(o))
        self.enable_disable.append(self.manual_single_assignment_add)
        single_vector_run_row.addWidget(self.manual_single_assignment_add)
        single_vector_layout.addLayout(single_vector_run_row)
        single_vector_tab.setLayout(single_vector_layout)
        assignment_creation_tabs.addTab(single_vector_tab, 'Single Vector')

        #Three Vector
        three_vector_tab = QWidget()
        three_vector_layout = QVBoxLayout()

        three_vector_feature_row = QHBoxLayout()
        three_vector_feature_sel_col = QVBoxLayout()
        label = QLabel('Feature')
        label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        three_vector_feature_sel_col.addWidget(label)
        self.manual_three_feature_combo = QComboBox()
        self.manual_three_feature_combo.activated.connect(self._on_three_feature_changed)
        three_vector_feature_sel_col.addWidget(self.manual_three_feature_combo)
        three_vector_feature_row.addLayout(three_vector_feature_sel_col)

        three_vector_include_col = QVBoxLayout()
        label = QLabel('Include')
        label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        three_vector_include_col.addWidget(label)
        self.manual_three_include_combo = CheckableComboBox()
        three_vector_include_col.addWidget(self.manual_three_include_combo)
        three_vector_feature_row.addLayout(three_vector_include_col)
        three_vector_layout.addLayout(three_vector_feature_row)

        three_vector_run_row = QHBoxLayout()
        self.manual_three_assignment_overwrite = QPushButton('Overwrite assignment vector(s)')
        self.manual_three_assignment_overwrite.clicked.connect(lambda _, o = True: self.on_manual_three_assignment_clicked(o))
        self.enable_disable.append(self.manual_three_assignment_overwrite)
        three_vector_run_row.addWidget(self.manual_three_assignment_overwrite)
        self.manual_three_assignment_add = QPushButton('Add assignment vector(s)')
        self.manual_three_assignment_add.clicked.connect(lambda _, o = False: self.on_manual_three_assignment_clicked(o))
        self.enable_disable.append(self.manual_three_assignment_add)
        three_vector_run_row.addWidget(self.manual_three_assignment_add)
        three_vector_layout.addLayout(three_vector_run_row)
        three_vector_tab.setLayout(three_vector_layout)
        assignment_creation_tabs.addTab(three_vector_tab, 'Three Vectors')

        weight_vector_tab = QWidget()
        weight_vector_layout = QVBoxLayout()

        self.weight_vectors = {}
        self.weight_vectors_included = []

        self.weight_vector_components_widget = QWidget()
        self.weight_vector_components = QVBoxLayout()
        self.weight_vector_components_widget.setLayout(self.weight_vector_components)

        v = QWidget()
        v.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        vector_component = QHBoxLayout()

        feature_col = QVBoxLayout()
        label = QLabel('Feature')
        label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        feature_col.addWidget(label)
        vector_component_feature_combo = QComboBox()
        vector_component_feature_combo.activated.connect(lambda _, i=0: self._on_component_feature_changed(i))
        feature_col.addWidget(vector_component_feature_combo)
        self.weight_vectors[0] = {'feature_combo': vector_component_feature_combo}
        self.weight_vectors_included.append(0)
        vector_component.addLayout(feature_col)

        include_col = QVBoxLayout()
        label = QLabel('Include')
        label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        include_col.addWidget(label)
        vector_component_include_combo = CheckableComboBox()
        include_col.addWidget(vector_component_include_combo)
        self.weight_vectors[0]['include_combo'] = vector_component_include_combo
        vector_component.addLayout(include_col)

        weight_add_col = QVBoxLayout()
        weight_row = QHBoxLayout()
        label = QLabel('Weight')
        label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        weight_row.addWidget(label)
        weight_box = QDoubleSpinBox()
        weight_box.setMinimum(0)
        weight_row.addWidget(weight_box)
        self.weight_vectors[0]['weight'] = weight_box
        weight_add_col.addLayout(weight_row)
        add_row_button = QPushButton('Add Vector')
        add_row_button.clicked.connect(self._on_vector_add_row)
        weight_add_col.addWidget(add_row_button)
        vector_component.addLayout(weight_add_col)
        v.setLayout(vector_component)

        self.weight_vector_components.addWidget(v)
        self.weight_vector_components_widget.setLayout(self.weight_vector_components)
        weight_vector_layout.addWidget(self.weight_vector_components_widget)

        weight_vector_name_row = QHBoxLayout()
        label = QLabel('Name:')
        label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        weight_vector_name_row.addWidget(label)
        self.manual_weight_vector_name = QLineEdit()
        self.manual_weight_vector_name.setText(self.current_manual_settings.get('vector_name', 'name'))
        weight_vector_name_row.addWidget(self.manual_weight_vector_name)
        weight_vector_layout.addLayout(weight_vector_name_row)

        weight_vector_run_row = QHBoxLayout()
        self.manual_weight_assignment_overwrite = QPushButton('Overwrite assignment vector(s)')
        self.manual_weight_assignment_overwrite.clicked.connect(lambda _, o = True: self.on_manual_weight_assignment_clicked(o))
        self.enable_disable.append(self.manual_weight_assignment_overwrite)
        weight_vector_run_row.addWidget(self.manual_weight_assignment_overwrite)
        self.manual_weight_assignment_add = QPushButton('Add assignment vector(s)')
        self.manual_weight_assignment_add.clicked.connect(lambda _, o = False: self.on_manual_weight_assignment_clicked(o))
        self.enable_disable.append(self.manual_weight_assignment_add)
        weight_vector_run_row.addWidget(self.manual_weight_assignment_add)

        weight_vector_layout.addLayout(weight_vector_run_row)
        weight_vector_tab.setLayout(weight_vector_layout)
        assignment_creation_tabs.addTab(weight_vector_tab, 'Weighted Vector')

        manual_assignment_form.addWidget(assignment_creation_tabs)

        manual_assignment.setLayout(manual_assignment_form)
        manual.addTab(manual_assignment, 'Assignment Vectors')
        
        ###
        # Manual Exposure Calculation
        ###
        manual_calculate = QWidget()
        self.manual_calculate_form = QVBoxLayout()

        # PDB file selection
        manual_calculate_file_row = QHBoxLayout()
        self.manual_calculate_file_edit = QLineEdit()
        self.manual_calculate_file_edit.setText(self.current_manual_settings.get('preprocessed_path_calculate', ''))
        self.manual_calculate_file_browse = QPushButton('Browse...')
        self.manual_calculate_file_browse.clicked.connect(self._browse_manual_calculate_file)
        manual_calculate_file_row.addWidget(QLabel('PDB/mmCIF File:'))
        manual_calculate_file_row.addWidget(self.manual_calculate_file_edit)
        manual_calculate_file_row.addWidget(self.manual_calculate_file_browse)
        self.manual_calculate_form.addLayout(manual_calculate_file_row)

        # Folder selection
        manual_calculate_folder_row = QHBoxLayout()
        self.manual_calculate_folder_edit = QLineEdit()
        self.manual_calculate_folder_edit.setText(self.current_manual_settings.get('calculate_folder_path', ''))
        self.manual_calculate_folder_browse = QPushButton('Browse...')
        self.manual_calculate_folder_browse.clicked.connect(self._browse_manual_calculate_folder)
        manual_calculate_folder_row.addWidget(QLabel('Output Folder:'))
        manual_calculate_folder_row.addWidget(self.manual_calculate_folder_edit)
        manual_calculate_folder_row.addWidget(self.manual_calculate_folder_browse)
        self.manual_calculate_form.addLayout(manual_calculate_folder_row)

        # Scoring Function
        self.functions = {}
        self.manual_functions_widget = QWidget()
        self.manual_functions = QVBoxLayout()
        self.manual_functions_widget.setLayout(self.manual_functions)

        m = QWidget()
        m.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        manual_function = QHBoxLayout()
        function_type_col = QVBoxLayout()
        label = QLabel('Function Type')
        label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        function_type_col.addWidget(label)
        combo = QComboBox()
        combo.addItems(self.current_manual_settings.get('function_types'))
        op = self.current_manual_settings.get('function_selected', 'Close and far cutoff')
        idx = combo.findText(op)
        if idx>=0:
            combo.setCurrentIndex(idx)
        function_type_col.addWidget(combo)
        manual_function.addLayout(function_type_col)
        self.functions[0] = {'function': combo}
        combo.currentIndexChanged.connect(lambda _, w=m: self._on_manual_reset_row(w, 0))

        self.manual_function_values = {}
        for constant, cvalue in self.current_manual_settings.get('function')['constants'].items():
            function_value_col = QVBoxLayout()
            label = QLabel(constant)
            label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            function_value_col.addWidget(label)
            value = QDoubleSpinBox()
            value.setValue(cvalue)
            function_value_col.addWidget(value)
            manual_function.addLayout(function_value_col)
            self.functions[0][constant] = value

        function_max_score_col = QVBoxLayout()
        label=QLabel('Max Score')
        label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        function_max_score_col.addWidget(label)
        max_score = QDoubleSpinBox()
        max_score.setMinimum(0)
        max_score.setMaximum(1e6)
        if self.weight_by_atomic_mass:
            max_score.setValue(self.current_manual_settings.get('function').get('max_score').get('weight_by_amu', 0))
        else:
            max_score.setValue(self.current_manual_settings.get('function').get('max_score').get('unweighted', 0))
        function_max_score_col.addWidget(max_score)
        manual_function.addLayout(function_max_score_col)
        self.functions[0]['max_score'] = max_score

        button_col = QVBoxLayout()
        reset_row_button = QPushButton('Calc Max Score')
        reset_row_button.clicked.connect(lambda _, w=m: self._on_calculate_maximum_score_clicked(w, 0))
        button_col.addWidget(reset_row_button)
        add_row_button = QPushButton('Add Function')
        add_row_button.clicked.connect(self._on_manual_add_row)
        button_col.addWidget(add_row_button)
        manual_function.addLayout(button_col)
        m.setLayout(manual_function)

        self.manual_functions.addWidget(m)

        self.manual_functions_widget.setLayout(self.manual_functions)

        self.manual_calculate_form.addWidget(self.manual_functions_widget)

        # Checkboxes
        manual_calculate_check_row = QHBoxLayout()
        self.calculate_assignment_checkbox = QCheckBox('Use generated assignment vectors?')
        self.calculate_assignment_checkbox.setChecked(self.calculate_assignment)
        self.calculate_assignment_checkbox.stateChanged.connect(self._on_calculate_assignment_toggled)
        manual_calculate_check_row.addWidget(self.calculate_assignment_checkbox)
        self.calculate_average_checkbox = QCheckBox('Average output per residue?')
        self.calculate_average_checkbox.setChecked(self.calculate_average)
        self.calculate_average_checkbox.stateChanged.connect(self._on_calculate_average_toggled)
        manual_calculate_check_row.addWidget(self.calculate_average_checkbox)
        self.calculate_backbone_checkbox = QCheckBox('Also average only by backbone?')
        self.calculate_backbone_checkbox.setChecked(self.calculate_backbone)
        self.calculate_backbone_checkbox.stateChanged.connect(self._on_calculate_backbone_toggled)
        manual_calculate_check_row.addWidget(self.calculate_backbone_checkbox)
        self.manual_calculate_form.addLayout(manual_calculate_check_row)

        # Calculate button
        self.run_manual_calculate = QPushButton('Calculate')
        self.run_manual_calculate.clicked.connect(self.on_run_manual_calculate_clicked)
        self.enable_disable.append(self.run_manual_calculate)
        self.manual_calculate_form.addWidget(self.run_manual_calculate)

        manual_calculate.setLayout(self.manual_calculate_form)
        manual.addTab(manual_calculate, 'Exposure Calculation')

        manual_form.addWidget(manual)

        self.manual_output = QTextEdit()
        self.manual_output.setReadOnly(True)
        self.manual_output.setPlaceholderText('Results will appear here...')
        manual_form.addWidget(self.manual_output)

        manual_tot.setLayout(manual_form)
        tabs.addTab(manual_tot, 'Manual')

        self.setCentralWidget(tabs)

    def weight_by_atomic_mass_clicked(self, state):
        self.weight_by_atomic_mass = bool(state)
        for sett in self.all_settings:
            sett['weight_by_atomic_mass'] = self.weight_by_atomic_mass

    def pickle_preprocessed_clicked(self, state):
        self.pickle_preprocessed = bool(state)
        for sett in self.all_settings:
            sett['pickle_preprocessed'] = self.pickle_preprocessed

    def on_run_simple_clicked(self):
        # gather values
        updated_settings = self.get_simple_settings()
        for k in updated_settings:
            self.current_simple_settings[k] = updated_settings.get(k)

        # disable buttons while running
        self.file_browse.setEnabled(False)
        self.folder_pre_browse.setEnabled(False)
        self.folder_out_browse.setEnabled(False)
        for b in self.enable_disable:
            b.setEnabled(False)

        # create worker & thread
        self.thread = QThread()
        self.worker = ScriptWorker(settings=self.current_simple_settings.copy())
        self.worker.moveToThread(self.thread)

        # connect signals
        self.thread.started.connect(self.worker.run_simple)
        self.worker.progress.connect(self.simple_output.append)
        self.worker.finished.connect(self.on_worker_simple_finished)
        self.worker.error.connect(self.on_worker_simple_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self.thread.quit)
        self.worker.error.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.worker.ask.connect(self._on_worker_ask_question)

        # start the thread
        self.thread.start()

    def on_worker_simple_finished(self, result):
        # re-enable run button
        self.file_browse.setEnabled(True)
        self.folder_pre_browse.setEnabled(True)
        self.folder_out_browse.setEnabled(True)
        for b in self.enable_disable:
            b.setEnabled(True)

        for i in result:
            self.simple_output.append(f'File {i[0]} saved. \n Min: {i[1]:.2f} \n Max: {i[2]:.2f}')
        # print(f'{result[0][0]}')
        self.visuals_pdb_edit.setText(f'{result[0][0]}')
        self._render_embed()

    def on_worker_simple_error(self, err_str):
        self.file_browse.setEnabled(True)
        self.folder_pre_browse.setEnabled(True)
        self.folder_out_browse.setEnabled(True)
        for b in self.enable_disable:
            b.setEnabled(True)
        self.simple_output.append(f'Worker error: {err_str}')
        QMessageBox.critical(self, 'Script error', f'An error occurred:\n{err_str}')

    def on_run_adduct_pre_clicked(self):
        # gather values
        updated_settings = self.get_adduct_settings()
        for k in updated_settings:
            self.current_adduct_settings[k] = updated_settings.get(k)

        # disable run button while running
        self.adduct_file_browse.setEnabled(False)
        self.adduct_folder_pre_browse.setEnabled(False)
        self.adduct_folder_out_browse.setEnabled(False)
        for b in self.enable_disable:
            b.setEnabled(False)

        # create worker & thread
        self.thread = QThread()
        self.worker = ScriptWorker(settings=self.current_adduct_settings.copy())
        self.worker.moveToThread(self.thread)

        # connect signals
        self.thread.started.connect(self.worker.run_adduct_pre)
        self.worker.progress.connect(self.adduct_output.append)
        self.worker.finished.connect(self.on_worker_adduct_pre_finished)
        self.worker.error.connect(self.on_worker_adduct_pre_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self.thread.quit)
        self.worker.error.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.worker.ask.connect(self._on_worker_ask_question)

        # start the thread
        self.thread.start()

    def on_worker_adduct_pre_finished(self, result):
        # re-enable run button
        self.adduct_file_browse.setEnabled(True)
        self.adduct_folder_pre_browse.setEnabled(True)
        self.adduct_folder_out_browse.setEnabled(True)
        for b in self.enable_disable:
            b.setEnabled(True)  

        pre_out, options = result
        self.current_adduct_settings['pre_out_path'] = pre_out
        self.adduct_output.append(f"File {pre_out} saved. \n There were {len(options)} unique entries under {self.current_adduct_settings.get('feature')}.")
        self.combo.clear()
        self.combo.addItems(options)

    def on_worker_adduct_pre_error(self, err_str):
        self.adduct_file_browse.setEnabled(True)
        self.adduct_folder_pre_browse.setEnabled(True)
        self.adduct_folder_out_browse.setEnabled(True)
        for b in self.enable_disable:
            b.setEnabled(True)  
        QMessageBox.critical(self, 'Script error', f'An error occurred:\n{err_str}')

    def on_run_adduct_out_clicked(self):
        # gather values
        updated_settings = self.get_adduct_settings()
        if updated_settings['combo'] == []:
            self.adduct_output.append('You much select at least one entry under Combo')
        else:
            for k in updated_settings:
                self.current_adduct_settings[k] = updated_settings.get(k)


            # disable run button while running
            self.adduct_file_browse.setEnabled(False)
            self.adduct_folder_pre_browse.setEnabled(False)
            self.adduct_folder_out_browse.setEnabled(False)
            for b in self.enable_disable:
                b.setEnabled(False)

            # create worker & thread
            self.thread = QThread()
            self.worker = ScriptWorker(settings=self.current_adduct_settings.copy())
            self.worker.moveToThread(self.thread)

            # connect signals
            self.thread.started.connect(self.worker.run_adduct_out)
            self.worker.progress.connect(self.adduct_output.append)
            self.worker.finished.connect(self.on_worker_adduct_out_finished)
            self.worker.error.connect(self.on_worker_adduct_out_error)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.worker.error.connect(self.thread.quit)
            self.worker.error.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            
            self.worker.ask.connect(self._on_worker_ask_question)

            # start the thread
            self.thread.start()

    def on_worker_adduct_out_finished(self, result):
        # re-enable run button
        self.adduct_file_browse.setEnabled(True)
        self.adduct_folder_pre_browse.setEnabled(True)
        self.adduct_folder_out_browse.setEnabled(True)    
        for b in self.enable_disable:
            b.setEnabled(True)
        for i in result:
            self.adduct_output.append(f'File {i[0]} saved. \n Min: {i[1]:.2f} \n Max: {i[2]:.2f}')
        self.visuals_pdb_edit.setText(f'{result[0][0]}')
        self._render_embed()

    def on_worker_adduct_out_error(self, err_str):
        self.adduct_file_browse.setEnabled(True)
        self.adduct_folder_pre_browse.setEnabled(True)
        self.adduct_folder_out_browse.setEnabled(True)    
        for b in self.enable_disable:
            b.setEnabled(True)
        QMessageBox.critical(self, 'Script error', f'An error occurred:\n{err_str}')

    def _on_simple_average_toggled(self, state):
        self.simple_average = bool(state)
        self.current_simple_settings['average'] = self.simple_average

    def _on_simple_backbone_toggled(self, state):
        self.simple_backbone = bool(state)
        self.current_simple_settings['backbone'] = self.simple_backbone

    def _on_adduct_average_toggled(self, state):
        self.adduct_average = bool(state)
        self.current_adduct_settings['average'] = self.adduct_average

    def _on_adduct_backbone_toggled(self, state):
        self.adduct_backbone = bool(state)
        self.current_adduct_settings['backbone'] = self.adduct_backbone

    def _on_only_chains_toggled(self, state):
        self.only_chain = bool(state)
        self.current_plot_settings['only_chain'] = self.only_chain

    def _on_only_backbones_toggled(self, state):
        self.only_backbone = bool(state)
        self.current_plot_settings['only_backbone'] = self.only_backbone

    def on_run_plot_clicked(self):
        # gather values
        updated_settings = self.get_plot_settings()
        self.plot_output.append(f'{updated_settings}')
        for k in updated_settings:
            self.current_plot_settings[k] = updated_settings.get(k)

        # disable run button while running
        for b in self.enable_disable:
            b.setEnabled(False)

        # create worker & thread
        self.thread = QThread()
        self.worker = ScriptWorker(settings=self.current_plot_settings.copy())
        self.worker.moveToThread(self.thread)

        # connect signals
        self.thread.started.connect(self.worker.run_plot)
        self.worker.finished.connect(self.on_worker_plot_finished)
        self.worker.error.connect(self.on_worker_plot_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self.thread.quit)
        self.worker.error.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.worker.ask.connect(self._on_worker_ask_question)

        # start the thread
        self.thread.start()

    def on_worker_plot_finished(self, result):
        # re-enable run button
        for b in self.enable_disable:
            b.setEnabled(True)

        # result is the dict emitted by worker
        if isinstance(result, dict):
            x = result['x']
            y = result['y']
            names = result.get('names', None)
            xlabel = result.get('xlabel', None)
            ylabel = result.get('ylabel', None)

            try:
                self.sc.plot_score_vs_resolution(x, y, names=names, xlabel=xlabel, ylabel=ylabel)
            except Exception as e:
                QMessageBox.critical(self, 'Plot error', f'Could not draw plot:\n{e}')
        else:
            # fallback text output if something unexpected
            self.plot_output.append('Plot worker finished (unexpected result format).')

    def on_worker_plot_error(self, err_str):
        for b in self.enable_disable:
            b.setEnabled(True)
        self.plot_output.append(f'Worker error: {err_str}')
        QMessageBox.critical(self, 'Script error', f'An error occurred:\n{err_str}')

    def _browse_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select file', self.file_edit.text() or '', 'All Files (*)')
        if fname:
            self.file_edit.setText(fname)

    def _browse_adduct_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select file', self.adduct_file_edit.text() or '', 'All Files (*)')
        if fname:
            self.adduct_file_edit.setText(fname)

    def _browse_plot_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select file', self.plot_pdb_edit.text() or '', 'All Files (*)')
        if fname:
            self.plot_pdb_edit.setText(fname)

    def _browse_defattr_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select file', self.plot_defattr_edit.text() or '', 'All Files (*)')
        if fname:
            self.plot_defattr_edit.setText(fname)

    def _browse_pre_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select folder', self.folder_pre_edit.text() or '')
        if folder:
            self.folder_pre_edit.setText(folder)
            
    def _browse_out_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select folder', self.folder_out_edit.text() or '')
        if folder:
            self.folder_out_edit.setText(folder)    
    
    def _browse_adduct_pre_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select folder', self.adduct_folder_pre_edit.text() or '')
        if folder:
            self.adduct_folder_pre_edit.setText(folder)
            
    def _browse_adduct_out_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select folder', self.adduct_folder_out_edit.text() or '')
        if folder:
            self.adduct_folder_out_edit.setText(folder)

    def _on_worker_ask_question(self, text):
        # This runs on the main thread (slot invoked in main thread)
        dlg = QMessageBox(self)
        dlg.setWindowTitle('User Input Required')
        dlg.setText(text)
        dlg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        dlg.setDefaultButton(QMessageBox.StandardButton.Yes)
        dlg.setIcon(QMessageBox.Icon.Question)
        button = dlg.exec()
        yn = (button == QMessageBox.StandardButton.Yes)
        # send the answer back to the worker
        # worker.answer is a signal defined in worker; safe to emit from main thread
        self.worker.answer.emit(yn)
        
    def _browse_visuals_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select PDB or mmCIF', basedir, "PDB/mmCIF Files (*.pdb *.pdb.gz *.ent *.cif *.cif.gz)")
        if fname:
            self.visuals_pdb_edit.setText(fname)

    def _render_embed(self):
        pdb_path = self.visuals_pdb_edit.text().strip()
        # print("[DEBUG] Render button clicked. pdb_path:", pdb_path, "python:", sys.executable)
        if not pdb_path:
            QMessageBox.warning(self, "No file", "Please select a PDB or mmCIF file first.")
            return
        if not Path(pdb_path).exists():
            QMessageBox.critical(self, "File not found", f"Could not find file:\n{pdb_path}")
            return

        try:
            fig = visualize(pdb_path=pdb_path)
            if fig is None:
                raise RuntimeError("visualize() returned None")
        except Exception as e:
            tb = traceback.format_exc()
            # print("[ERROR] Exception while building figure:\n", tb)
            QMessageBox.critical(self, "Visualization error", f"Could not build figure:\n{e}\n\nSee console for traceback.")
            return

        if not WEBENGINE_AVAILABLE:
            QMessageBox.information(self, "No WebEngine", "Qt WebEngine not available; use 'Open in Browser' instead.")
            return

        try:
            # Use full_html=True so we deliver a complete HTML doc to the QWebEngineView.
            html = fig.to_html(include_plotlyjs='inline', full_html=True)
            self._show_html_in_file(html)
            # print("[DEBUG] setHtml called and reload requested.")
        except Exception as e:
            tb = traceback.format_exc()
            # print("[ERROR] Exception while embedding HTML:\n", tb)
            QMessageBox.critical(self, "Embed error", f"Could not embed figure:\n{e}\n\nSee console for traceback.")

    def _render_plot_embed(self):
        updated_settings = self.get_plot_settings()
        for k in updated_settings:
            self.current_plot_settings[k] = updated_settings.get(k)

        pdb_path = self.current_plot_settings.get('pdb_path')
        defattr_path = self.current_plot_settings.get('defattr_path')
        only_chain = self.current_plot_settings.get('only_chain')
        only_backbone = self.current_plot_settings.get('only_backbone')

        if not pdb_path:
            QMessageBox.warning(self, "No file", "Please select a PDB or mmCIF file first.")
            return
        if not Path(pdb_path).exists():
            QMessageBox.critical(self, "File not found", f"Could not find file:\n{pdb_path}")
            return

        if not defattr_path:
            QMessageBox.warning(self, "No file", "Please select a defattr file first.")
            return
        if not Path(defattr_path).exists():
            QMessageBox.critical(self, "File not found", f"Could not find file:\n{defattr_path}")
            return
        
        try:
            fig = score_v_localres_plotly(pdb_path=pdb_path, defattr_path=defattr_path, only_chain=only_chain, backboneonly=only_backbone)
            if fig is None:
                raise RuntimeError("score_v_localres_plotly() returned None")
        except Exception as e:
            tb = traceback.format_exc()
            # print("[ERROR] Exception while building figure:\n", tb)
            QMessageBox.critical(self, "Visualization error", f"Could not build figure:\n{e}\n\nSee console for traceback.")
            return

        if not WEBENGINE_AVAILABLE:
            QMessageBox.information(self, "No WebEngine", "Qt WebEngine not available; use 'Open in Browser' instead.")
            return

        try:
            # Use full_html=True so we deliver a complete HTML doc to the QWebEngineView.
            html = fig.to_html(include_plotlyjs='inline', full_html=True)
            self._show_plot_html_in_file(html)
        except Exception as e:
            tb = traceback.format_exc()
            print("[ERROR] Exception while embedding HTML:\n", tb)
            QMessageBox.critical(self, "Embed error", f"Could not embed figure:\n{e}\n\nSee console for traceback.")

    def _show_html_in_file(self, html: str):
        tmpdir = tempfile.gettempdir()
        fname = f"plot_{uuid4().hex}.html"
        path = os.path.join(tmpdir, fname)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        # load as file URL so a fresh document is created
        self.visuals_view.setUrl(QUrl.fromLocalFile(path))

    def _show_plot_html_in_file(self, html: str):
        tmpdir = tempfile.gettempdir()
        fname = f"plot_{uuid4().hex}.html"
        path = os.path.join(tmpdir, fname)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        # load as file URL so a fresh document is created
        self.plot_view.setUrl(QUrl.fromLocalFile(path))

    def _browse_manual_preprocess_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select file', self.manual_preprocess_file_edit.text() or '', 'All Files (*)')
        if fname:
            self.manual_preprocess_file_edit.setText(fname)

    def _browse_manual_preprocess_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select folder', self.manual_preprocess_folder_edit.text() or '')
        if folder:
            self.manual_preprocess_folder_edit.setText(folder)

    def _on_preprocess_redefine_chains_toggled(self, state):
        self.preprocess_redefine_chains = bool(state)
        self.current_manual_settings['preprocess_redefine_chains'] = self.preprocess_redefine_chains

    def on_run_manual_preprocess_clicked(self):
        # gather values
        updated_settings = self.get_manual_settings()
        for k in updated_settings:
            self.current_manual_settings[k] = updated_settings.get(k)

        # disable run button while running
        self.manual_preprocess_file_browse.setEnabled(False)
        self.manual_preprocess_folder_browse.setEnabled(False)
        self.preprocess_redefine_chains_checkbox.setEnabled(False)
        for b in self.enable_disable:
            b.setEnabled(False)

        # create worker & thread
        self.thread = QThread()
        self.worker = ScriptWorker(settings=self.current_manual_settings.copy())
        self.worker.moveToThread(self.thread)

        # connect signals
        self.thread.started.connect(self.worker.run_manual_preprocessing)
        self.worker.progress.connect(self.manual_output.append)
        self.worker.finished.connect(self.on_worker_manual_preprocess_finished)
        self.worker.error.connect(self.on_worker_manual_preprocess_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self.thread.quit)
        self.worker.error.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.ask.connect(self._on_worker_ask_question)

        # start the thread
        self.thread.start()

    def on_worker_manual_preprocess_finished(self, result):
        # re-enable run button
        self.manual_preprocess_file_browse.setEnabled(True)
        self.manual_preprocess_folder_browse.setEnabled(True)
        self.preprocess_redefine_chains_checkbox.setEnabled(True)
        for b in self.enable_disable:
            b.setEnabled(True)
        self.current_manual_settings['preprocessed_path_assignment'] = result
        self.current_manual_settings['preprocessed_path_calculate'] = result
        self.manual_calculate_file_edit.setText(result)
        self.manual_assignment_file_edit.setText(result)
        self.manual_single_feature_combo.clear()
        self.manual_single_feature_combo.addItems(getcols(result, True))
        self.manual_three_feature_combo.clear()
        self.manual_three_feature_combo.addItems(getcols(result, True))
        for key, value in self.weight_vectors.items():
                value['feature_combo'].clear()
                value['feature_combo'].addItems(getcols(result, True))
        self.manual_output.append(f"Preprocessing complete. File {result} saved.")

    def on_worker_manual_preprocess_error(self, err_str):
        self.manual_preprocess_file_browse.setEnabled(True)
        self.manual_preprocess_folder_browse.setEnabled(True)
        self.preprocess_redefine_chains_checkbox.setEnabled(True)
        for b in self.enable_disable:
            b.setEnabled(True)
        QMessageBox.critical(self, 'Script error', f'An error occurred:\n{err_str}')



    def on_manual_assignment_reset_clicked(self):
        dlg = QMessageBox(self)
        dlg.setWindowTitle('User Input Required')
        dlg.setText('Are you sure you want to clear all assignment vectors?\nThis cannot be undone.')
        dlg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        dlg.setDefaultButton(QMessageBox.StandardButton.Yes)
        dlg.setIcon(QMessageBox.Icon.Question)
        button = dlg.exec()
        yn = (button == QMessageBox.StandardButton.Yes)
        if yn:
            self.current_manual_settings['assignment_vectors'] = None
            self.manual_output.append('Assignment vectors cleared')

    def _browse_manual_assignment_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select file', self.manual_assignment_file_edit.text() or '', 'All Files (*)')
        if fname:
            self.manual_assignment_file_edit.setText(fname)
            self.manual_calculate_file_edit.setText(fname)
            self.manual_single_feature_combo.clear()
            self.manual_single_feature_combo.addItems(getcols(fname, True))
            self.manual_three_feature_combo.clear()
            self.manual_three_feature_combo.addItems(getcols(fname, True))
            for key, value in self.weight_vectors.items():
                value['feature_combo'].clear()
                value['feature_combo'].addItems(getcols(fname, True))

    def _on_single_feature_changed(self):
        self.manual_single_include_combo.clear()
        tempfeature = features(self.manual_assignment_file_edit.text(), feature = self.manual_single_feature_combo.currentText(), yn=True)
        strfeature = []
        for i in tempfeature:
            strfeature.append(str(i))
        self.manual_single_include_combo.addItems(strfeature)

    def on_manual_single_assignment_clicked(self, overwrite):
        if overwrite:
            dlg = QMessageBox(self)
            dlg.setWindowTitle('User Input Required')
            dlg.setText('Are you sure you want to overwrite all assignment vectors?\nThis cannot be undone.')
            dlg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            dlg.setDefaultButton(QMessageBox.StandardButton.Yes)
            dlg.setIcon(QMessageBox.Icon.Question)
            button = dlg.exec()
            yn = (button == QMessageBox.StandardButton.Yes)
            if not yn:
                return

        updated_settings = self.get_manual_settings()
        for k in updated_settings:
            self.current_manual_settings[k] = updated_settings.get(k)

        if self.current_manual_settings.get('preprocessed_path_assignment') == '' or self.current_manual_settings.get('single_include') == '' or self.current_manual_settings.get('single_feature') == '':
            self.manual_output.append('Inputs not selected.')
            return

        # disable run button while running
        self.manual_single_include_combo.setEnabled(False)
        self.manual_single_feature_combo.setEnabled(False)
        self.manual_assignment_file_edit.setEnabled(False)
        for b in self.enable_disable:
            b.setEnabled(False)

        result = create_vectors(pdb_path=self.current_manual_settings.get('preprocessed_path_assignment'), 
                                include=self.current_manual_settings.get('single_include'),
                                feature=self.current_manual_settings.get('single_feature'))
        
        if overwrite:
            self.current_manual_settings['assignment_vectors'] = result
        elif self.current_manual_settings.get('assignment_vectors') == None:
            self.current_manual_settings['assignment_vectors'] = result
        else:
            for ke, va in result.items():
                self.current_manual_settings['assignment_vectors'][ke] = va
        
        self.manual_output.append(f"{len(self.current_manual_settings.get('assignment_vectors'))} assignment vector(s): {list(self.current_manual_settings.get('assignment_vectors').keys())}")

        self.manual_single_include_combo.setEnabled(True)
        self.manual_single_feature_combo.setEnabled(True)
        self.manual_assignment_file_edit.setEnabled(True)
        for b in self.enable_disable:
            b.setEnabled(True)



    def _on_three_feature_changed(self):
        self.manual_three_include_combo.clear()
        tempfeature = features(self.manual_assignment_file_edit.text(), feature = self.manual_three_feature_combo.currentText(), yn=True)
        strfeature = []
        for i in tempfeature:
            strfeature.append(str(i))
        self.manual_three_include_combo.addItems(strfeature)

    def on_manual_three_assignment_clicked(self, overwrite):
        if overwrite:
            dlg = QMessageBox(self)
            dlg.setWindowTitle('User Input Required')
            dlg.setText('Are you sure you want to overwrite all assignment vectors?\nThis cannot be undone.')
            dlg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            dlg.setDefaultButton(QMessageBox.StandardButton.Yes)
            dlg.setIcon(QMessageBox.Icon.Question)
            button = dlg.exec()
            yn = (button == QMessageBox.StandardButton.Yes)
            if not yn:
                return

        updated_settings = self.get_manual_settings()
        for k in updated_settings:
            self.current_manual_settings[k] = updated_settings.get(k)

        if self.current_manual_settings.get('preprocessed_path_assignment') == '' or self.current_manual_settings.get('three_include') == '' or self.current_manual_settings.get('three_feature') == '':
            self.manual_output.append('Inputs not selected.')
            return

        # disable run button while running
        self.manual_three_include_combo.setEnabled(False)
        self.manual_three_feature_combo.setEnabled(False)
        self.manual_assignment_file_edit.setEnabled(False)
        for b in self.enable_disable:
            b.setEnabled(False)

        result = create_3_vectors(pdb_path=self.current_manual_settings.get('preprocessed_path_assignment'), 
                                  chain1=self.current_manual_settings.get('three_include'),
                                  feature=self.current_manual_settings.get('three_feature'),)
        

        if overwrite:
            self.current_manual_settings['assignment_vectors'] = result
        elif self.current_manual_settings.get('assignment_vectors') == None:
            self.current_manual_settings['assignment_vectors'] = result
        else:
            for ke, va in result.items():
                self.current_manual_settings['assignment_vectors'][ke] = va
        
        self.manual_output.append(f"{len(self.current_manual_settings.get('assignment_vectors'))} assignment vector(s): {list(self.current_manual_settings.get('assignment_vectors').keys())}")

        self.manual_three_include_combo.setEnabled(True)
        self.manual_three_feature_combo.setEnabled(True)
        self.manual_assignment_file_edit.setEnabled(True)
        for b in self.enable_disable:
            b.setEnabled(True)


    def _on_component_feature_changed(self, index):
        self.weight_vectors[index]['include_combo'].clear()
        tempfeature = features(self.manual_assignment_file_edit.text(), feature = self.weight_vectors[index]['feature_combo'].currentText(), yn=True)
        strfeature = []
        for i in tempfeature:
            strfeature.append(str(i))
        self.weight_vectors[index]['include_combo'].addItems(strfeature)

    def on_manual_weight_assignment_clicked(self, overwrite):
        if overwrite:
            dlg = QMessageBox(self)
            dlg.setWindowTitle('User Input Required')
            dlg.setText('Are you sure you want to overwrite all assignment vectors?\nThis cannot be undone.')
            dlg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            dlg.setDefaultButton(QMessageBox.StandardButton.Yes)
            dlg.setIcon(QMessageBox.Icon.Question)
            button = dlg.exec()
            yn = (button == QMessageBox.StandardButton.Yes)
            if not yn:
                return

        updated_settings = self.get_manual_settings()
        for k in updated_settings:
            self.current_manual_settings[k] = updated_settings.get(k)

        if self.current_manual_settings.get('preprocessed_path_assignment') == '' or self.current_manual_settings.get('three_include') == '' or self.current_manual_settings.get('three_feature') == '':
            self.manual_output.append('Inputs not selected.')
            return

        # disable run button while running
        self.manual_three_include_combo.setEnabled(False)
        self.manual_three_feature_combo.setEnabled(False)
        self.manual_assignment_file_edit.setEnabled(False)
        for b in self.enable_disable:
            b.setEnabled(False)

        for i, idx in enumerate(self.weight_vectors_included):
            feature = self.weight_vectors[idx]['feature_combo'].currentText()
            include = self.weight_vectors[idx]['include_combo'].currentData()
            weight = float(self.weight_vectors[idx]['weight'].text())
            if weight % 1 == 0:
                weight = int(weight)
            if i == 0:
                result = list(create_vectors(pdb_path=self.current_manual_settings.get('preprocessed_path_assignment'), 
                                             include=include,
                                             feature=feature).values())[0] * weight
            else:
                result = result + list(create_vectors(pdb_path=self.current_manual_settings.get('preprocessed_path_assignment'), 
                                                      include=include,
                                                      feature=feature).values())[0] * weight
        
        if overwrite:
            self.current_manual_settings['assignment_vectors'] = {self.current_manual_settings.get('vector_name'): result}
        elif self.current_manual_settings.get('assignment_vectors') == None:
            self.current_manual_settings['assignment_vectors'] = {self.current_manual_settings.get('vector_name'): result}
        else:
            self.current_manual_settings['assignment_vectors'][self.current_manual_settings.get('vector_name')] = result
        
        self.manual_output.append(f"{len(self.current_manual_settings.get('assignment_vectors'))} assignment vector(s): {list(self.current_manual_settings.get('assignment_vectors').keys())}")

        self.manual_three_include_combo.setEnabled(True)
        self.manual_three_feature_combo.setEnabled(True)
        self.manual_assignment_file_edit.setEnabled(True)
        for b in self.enable_disable:
            b.setEnabled(True)

    def _on_vector_remove_row(self, widget, tracking_index):
        if widget is not None:
            # Remove it from the layout and delete it

            self.weight_vector_components.removeWidget(widget)
            self.weight_vectors.pop(tracking_index)
            self.weight_vectors_included.pop(tracking_index)
            widget.deleteLater()

    def _on_vector_add_row(self):
        index = self.weight_vector_components.count()
        tracking_index = self.weight_vectors_included[-1] + 1

        v = QWidget()
        v.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        vector_component = QHBoxLayout()

        feature_col = QVBoxLayout()
        label = QLabel('Feature')
        label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        feature_col.addWidget(label)
        vector_component_feature_combo = QComboBox()
        vector_component_feature_combo.addItems(getcols(self.manual_assignment_file_edit.text(), True))
        vector_component_feature_combo.activated.connect(lambda _, i=tracking_index: self._on_component_feature_changed(i))
        feature_col.addWidget(vector_component_feature_combo)
        self.weight_vectors[tracking_index] = {'feature_combo': vector_component_feature_combo}
        self.weight_vectors_included.append(tracking_index)
        vector_component.addLayout(feature_col)

        include_col = QVBoxLayout()
        label = QLabel('Include')
        label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        include_col.addWidget(label)
        vector_component_include_combo = CheckableComboBox()
        include_col.addWidget(vector_component_include_combo)
        self.weight_vectors[tracking_index]['include_combo'] = vector_component_include_combo
        vector_component.addLayout(include_col)

        weight_add_col = QVBoxLayout()
        weight_row = QHBoxLayout()
        label = QLabel('Weight')
        label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        weight_row.addWidget(label)
        weight_box = QDoubleSpinBox()
        weight_box.setMinimum(0)
        weight_row.addWidget(weight_box)
        self.weight_vectors[tracking_index]['weight'] = weight_box
        weight_add_col.addLayout(weight_row)
        remove_row_button = QPushButton('Remove Vector')
        remove_row_button.clicked.connect(lambda _, w=v, i=tracking_index: self._on_vector_remove_row(w,i))
        weight_add_col.addWidget(remove_row_button)
        vector_component.addLayout(weight_add_col)
        v.setLayout(vector_component)

        self.weight_vector_components.insertWidget(index, v)



    def _browse_manual_calculate_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select file', self.manual_calculate_file_edit.text() or '', 'All Files (*)')
        if fname:
            self.manual_calculate_file_edit.setText(fname)

    def _browse_manual_calculate_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select folder', self.manual_calculate_folder_edit.text() or '')
        if folder:
            self.manual_calculate_folder_edit.setText(folder)

    def _on_calculate_assignment_toggled(self, state):
        self.calculate_assignment = bool(state)
        self.current_manual_settings['calculate_assignment'] = self.calculate_assignment

    def _on_calculate_average_toggled(self, state):
        self.calculate_average = bool(state)
        self.current_manual_settings['average'] = self.calculate_average

    def _on_calculate_backbone_toggled(self, state):
        self.calculate_backbone = bool(state)
        self.current_manual_settings['backbone'] = self.calculate_backbone

    def _on_manual_remove_row(self, widget, index):
        if widget is not None:
            # Remove it from the layout and delete it

            self.manual_functions.removeWidget(widget)
            widget.deleteLater()

    def _on_manual_reset_row(self, widget, index):
        if widget is not None:
            
            # containing_layout = widget.parent().layout()

            op = self.functions[index]['function'].currentText()

            m = QWidget()
            m.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            # containing_layout.replaceWidget(widget, m)
            manual_function = QHBoxLayout()
            function_type_col = QVBoxLayout()
            label = QLabel('Function Type')
            label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            function_type_col.addWidget(label)
            combo = QComboBox()
            combo.addItems(self.current_manual_settings.get('function_types'))
            idx = combo.findText(op)
            if idx>=0:
                combo.setCurrentIndex(idx)
            function_type_col.addWidget(combo)
            manual_function.addLayout(function_type_col)
            self.functions[index] = {'function': combo}
            combo.activated.connect(lambda _, w=m, i=index: self._on_manual_reset_row(w,i))

            self.manual_function_values = {}
            for constant, cvalue in available_scoring_functions.get(op)['constants'].items():
                function_value_col = QVBoxLayout()
                label = QLabel(constant)
                label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
                function_value_col.addWidget(label)
                value = QDoubleSpinBox()
                value.setValue(cvalue)
                function_value_col.addWidget(value)
                manual_function.addLayout(function_value_col)
                self.functions[index][constant] = value

            function_max_score_col = QVBoxLayout()
            label=QLabel('Max Score')
            label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            function_max_score_col.addWidget(label)
            max_score = QDoubleSpinBox()
            max_score.setMinimum(0)
            max_score.setMaximum(1e6)
            if self.weight_by_atomic_mass:
                max_score.setValue(self.current_manual_settings.get('function').get('max_score').get('weight_by_amu', 0))
            else:
                max_score.setValue(self.current_manual_settings.get('function').get('max_score').get('unweighted', 0))
            function_max_score_col.addWidget(max_score)
            manual_function.addLayout(function_max_score_col)
            self.functions[index]['max_score'] = max_score

            button_col = QVBoxLayout()
            reset_row_button = QPushButton('Calc Max Score')
            reset_row_button.clicked.connect(lambda _, w=m, i=index: self._on_calculate_maximum_score_clicked(w,i))
            button_col.addWidget(reset_row_button)
            if index == 0:
                add_row_button = QPushButton('Add Function')
                add_row_button.clicked.connect(self._on_manual_add_row)
                button_col.addWidget(add_row_button)
            else:
                remove_row_button = QPushButton('Remove Function')
                remove_row_button.clicked.connect(lambda _, w=m, i=index: self._on_manual_remove_row(w, i))
                button_col.addWidget(remove_row_button)
            manual_function.addLayout(button_col)
            m.setLayout(manual_function)

            self.manual_functions.replaceWidget(widget, m)
            self._on_manual_remove_row(widget, index)


    def _on_manual_add_row(self):
        index = self.manual_functions.count()

        m = QWidget()
        m.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        manual_function = QHBoxLayout()
        function_type_col = QVBoxLayout()
        label = QLabel('Function Type')
        label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        function_type_col.addWidget(label)
        combo = QComboBox()
        combo.addItems(self.current_manual_settings.get('function_types'))
        op = self.current_manual_settings.get('function_selected', 'Close and far cutoff')
        idx = combo.findText(op)
        if idx>=0:
            combo.setCurrentIndex(idx)
        function_type_col.addWidget(combo)
        manual_function.addLayout(function_type_col)
        self.functions[index] = {'function': combo}
        combo.currentIndexChanged.connect(lambda _, w=m, i=index: self._on_manual_reset_row(w,i))

        self.manual_function_values = {}
        for constant, cvalue in self.current_manual_settings.get('function')['constants'].items():
            function_value_col = QVBoxLayout()
            label = QLabel(constant)
            label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            function_value_col.addWidget(label)
            value = QDoubleSpinBox()
            value.setValue(cvalue)
            function_value_col.addWidget(value)
            manual_function.addLayout(function_value_col)
            self.functions[index][constant] = value

        function_max_score_col = QVBoxLayout()
        label=QLabel('Max Score')
        label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        function_max_score_col.addWidget(label)
        max_score = QDoubleSpinBox()
        max_score.setMinimum(0)
        max_score.setMaximum(1e6)
        if self.weight_by_atomic_mass:
            max_score.setValue(self.current_manual_settings.get('function').get('max_score').get('weight_by_amu', 0))
        else:
            max_score.setValue(self.current_manual_settings.get('function').get('max_score').get('unweighted', 0))
        function_max_score_col.addWidget(max_score)
        manual_function.addLayout(function_max_score_col)
        self.functions[index]['max_score'] = max_score

        button_col = QVBoxLayout()
        reset_row_button = QPushButton('Calc Max Score')
        reset_row_button.clicked.connect(lambda _, w=m, i=index: self._on_calculate_maximum_score_clicked(w,i))
        button_col.addWidget(reset_row_button)
        remove_row_button = QPushButton('Remove Function')
        remove_row_button.clicked.connect(lambda _, w=m, i=index: self._on_manual_remove_row(w,i))
        button_col.addWidget(remove_row_button)
        manual_function.addLayout(button_col)
        m.setLayout(manual_function)

        self.manual_functions.insertWidget(index, m)

    def _on_calculate_maximum_score_clicked(self, widget, index):
        updated_settings = self.get_manual_settings()
        for k in updated_settings:
            self.current_manual_settings[k] = updated_settings.get(k)

        tempconstants = {}
        for key, value in self.functions[index].items():
            if key == 'function':
                temp = available_scoring_functions[value.currentText()].copy()
            elif key == 'max_score':
                temp[key] = 0
            elif float(value.text()) % 1 == 0:
                tempconstants[key] = int(float(value.text()))
            else:
                tempconstants[key] = float(value.text())
            temp['constants'] = tempconstants.copy()

        ass=None

        if self.current_manual_settings.get('calculate_assignment'):
            dlg = QMessageBox(self)
            dlg.setWindowTitle('User Input Required')
            dlg.setText('Would you like to calculate the maximum score based on the current weighted vector?\nNote: Must have been prepared weighted vector using an mmCIF file.')
            dlg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            dlg.setDefaultButton(QMessageBox.StandardButton.No)
            dlg.setIcon(QMessageBox.Icon.Question)
            button = dlg.exec()
            yn = (button == QMessageBox.StandardButton.Yes)
            if yn:
                for i, idx in enumerate(self.weight_vectors_included):
                    feature = self.weight_vectors[idx]['feature_combo'].currentText()
                    include = self.weight_vectors[idx]['include_combo'].currentData()
                    weight = float(self.weight_vectors[idx]['weight'].text())
                    if weight % 1 == 0:
                        weight = int(weight)
                    if i == 0:
                        result = list(create_vectors(pdb_path=os.path.join(basedir, 'standards', 'rubisco.pkl'), 
                                                     include=include,
                                                     feature=feature).values())[0] * weight
                    else:
                        result = result + list(create_vectors(pdb_path=os.path.join(basedir, 'standards', 'rubisco.pkl'), 
                                                              include=include,
                                                              feature=feature).values())[0] * weight
                ass = {'weighted': result}


        df_out = max_exposure_score(funcs=temp, assignment=ass, subsample=1600, yn=self.yes_no, weight_by_amu=self.current_manual_settings.get('weight_by_atomic_mass'))

        self.functions[index]['max_score'].setValue(df_out.iloc[0,0])
        self.manual_output.append(f'Max score calculated: {df_out.iloc[0,0]:.2f}')
        if df_out.shape != (1,1):
            self.manual_output.append(f'{df_out}')
            self.manual_output.append('WARNING: multiple assignment vectors provided for max score calculation. First entry used to set max score.')

    def yes_no(self, text):
        dlg = QMessageBox(self)
        dlg.setWindowTitle('User Input Required')
        dlg.setText(text)
        dlg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        dlg.setDefaultButton(QMessageBox.StandardButton.Yes)
        dlg.setIcon(QMessageBox.Icon.Question)
        button = dlg.exec()
        yn = (button == QMessageBox.StandardButton.Yes)
        return yn


    def on_run_manual_calculate_clicked(self):
        # gather values
        updated_settings = self.get_manual_settings()
        for k in updated_settings:
            self.current_manual_settings[k] = updated_settings.get(k)
        
        if self.current_manual_settings.get('preprocessed_path_calculate') == '':
            self.manual_output.append('No pdb/cif file selected.')

        else:
            # disable run button while running
            self.manual_calculate_file_browse.setEnabled(False)
            self.manual_calculate_folder_browse.setEnabled(False)
            self.calculate_assignment_checkbox.setEnabled(False)
            for b in self.enable_disable:
                b.setEnabled(False)

            # create worker & thread
            self.thread = QThread()
            self.worker = ScriptWorker(settings=self.current_manual_settings.copy())
            self.worker.moveToThread(self.thread)

            # connect signals
            self.thread.started.connect(self.worker.run_manual_calculation)
            self.worker.progress.connect(self.manual_output.append)
            self.worker.finished.connect(self.on_worker_manual_calculate_finished)
            self.worker.error.connect(self.on_worker_manual_calculate_error)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.worker.error.connect(self.thread.quit)
            self.worker.error.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.worker.ask.connect(self._on_worker_ask_question)

            # start the thread
            self.thread.start()

    def on_worker_manual_calculate_finished(self, result):
        # re-enable run button
        self.manual_calculate_file_browse.setEnabled(True)
        self.manual_calculate_folder_browse.setEnabled(True)
        self.calculate_assignment_checkbox.setEnabled(True)
        for b in self.enable_disable:
            b.setEnabled(True)
        for i in result:
            self.manual_output.append(f'File {i[0]} saved. \n Min: {i[1]:.2f} \n Max: {i[2]:.2f}')
        self.visuals_pdb_edit.setText(f'{result[0][0]}')
        self._render_embed()

    def on_worker_manual_calculate_error(self, err_str):
        self.manual_calculate_file_browse.setEnabled(True)
        self.manual_calculate_folder_browse.setEnabled(True)
        self.calculate_assignment_checkbox.setEnabled(True)
        for b in self.enable_disable:
            b.setEnabled(True)
        QMessageBox.critical(self, 'Script error', f'An error occurred:\n{err_str}')





    def get_manual_settings(self):
        f = []
        for i in range(self.manual_functions.count()):
            tempconstants = {}
            for key, value in self.functions[i].items():
                if key == 'function':
                    temp = available_scoring_functions[value.currentText()].copy()
                elif key == 'max_score':
                    if self.weight_by_atomic_mass:
                        temp[key] = {'weight_by_amu': float(value.text())}
                    else:
                        temp[key] = {'unweighted': float(value.text())}
                elif float(value.text()) % 1 == 0:
                    tempconstants[key] = int(float(value.text()))
                else:
                    tempconstants[key] = float(value.text())
            temp['constants'] = tempconstants.copy()
            f.append(temp.copy())


        return {
            'preprocess_file_path': self.manual_preprocess_file_edit.text(),
            'preprocess_folder_path': self.manual_preprocess_folder_edit.text(),
            'preprocess_redefine_chains': self.preprocess_redefine_chains,
            'preprocessed_path_assignment': self.manual_assignment_file_edit.text(),
            'single_include': self.manual_single_include_combo.currentData(),
            'single_feature': self.manual_single_feature_combo.currentText(),
            'three_include': self.manual_three_include_combo.currentData(),
            'three_feature': self.manual_three_feature_combo.currentText(),
            'vector_name': self.manual_weight_vector_name.text(),
            'preprocessed_path_calculate': self.manual_calculate_file_edit.text(),
            'calculate_folder_path': self.manual_calculate_folder_edit.text(),
            'calculate_assignment': self.calculate_assignment,
            'average': self.calculate_average,
            'backbone': self.calculate_backbone,
            'funcs': f,
        }

    def get_simple_settings(self):
        # Return a dict of settings
        return {
            'pdb_path': self.file_edit.text(),
            'folder_pre_path': self.folder_pre_edit.text(),
            'folder_out_path': self.folder_out_edit.text(),
            'average': self.simple_average,
            'backbone': self.simple_backbone,
        }
    
    def get_adduct_settings(self):
        return {
            'pdb_path': self.adduct_file_edit.text(),
            'folder_pre_path': self.adduct_folder_pre_edit.text(),
            'folder_out_path': self.adduct_folder_out_edit.text(),
            'feature': self.adduct_feature.currentText(),
            'combo': self.combo.currentData(),
            'average': self.adduct_average,
            'backbone': self.adduct_backbone,
        }
    
    def get_plot_settings(self):
        if self.only_chain:
            return {
            'pdb_path': self.plot_pdb_edit.text(),
            'defattr_path': self.plot_defattr_edit.text(),
            'only_chain': self.only_chain_combo.currentData(),
            'only_backbone': self.only_backbone,
        }
        else:
            return {
            'pdb_path': self.plot_pdb_edit.text(),
            'defattr_path': self.plot_defattr_edit.text(),
            'only_backbone': self.only_backbone,
        }

app = QApplication(sys.argv)
app.setWindowIcon(QIcon(os.path.join(basedir, "standards", "icon.png")))

window = MainWindow()
window.show()

exit_code = 0
try:
    exit_code = app.exec()
finally:
    try:
        if window and hasattr(window, "visuals_view"):
            _cleanup_webview(window.visuals_view)
    except Exception:
        pass
    app.processEvents()
    sys.exit(exit_code)