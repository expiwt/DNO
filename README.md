# DNO - Diffeomorphic Neural Operator


**Diffeomorphic Neural Operator** - метод для решения PDE на семействах 2D-доменов с разной геометрией и топологией. Идея: через диффеоморфное отображение привести все геометрии к универсальному домену, обучить нейрооператор в нём, и обобщать на новые формы без переобучения.

---

## Целевые уравнения

### 1. Стационарное уравнение Дарси (диффузия в пористой среде)

$$-\nabla \cdot (c(x,y)\nabla u(x,y)) = F(x,y), \quad u|_{\partial \Omega}=0$$

- **Вход:** поле проницаемости $c(x,y)$ + описание геометрии (может быть с отверстием)
- **Выход:** поле давления $u(x,y)$
- **Данные:** FEM (MATLAB/Firedrake), CSV на сетке $128 \times 128$

### 2. Нестационарные уравнения Навье–Стокса (DNS + осреднение)

$$
\begin{cases}
\displaystyle\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} - \nu \nabla^2 \mathbf{u} + \nabla p = \mathbf{f}, \\
\nabla \cdot \mathbf{u} = 0, \\
\mathbf{u}(0) = \mathbf{u}_0,
\end{cases}
\quad \mathbf{u}|_{\partial \Omega} = \mathbf{u}_{\text{BC}}
$$

Решается методом конечных элементов (Firedrake) с **неявным Эйлером** по времени для турбулентного режима $\mathrm{Re} \in [800, 1500]$:
- Прогрев $T_{\text{burn}} = 20$ с (выход на статистически стационарный режим)
- Сбор статистики $T_{\text{sampling}} = 25$ с с шагом $\Delta t = 0.01$
- Time-averaging -> осреднённые поля $[\bar{u}, \bar{v}, \bar{p}]$

**Обучение FNO+FiLM:** нейрооператор учится предсказывать осреднённое поле по геометрии ($[\xi, \eta]$) и числу Рейнольдса (FiLM-кондиционирование).

- **Вход:** $[\text{Mask}, \xi, \eta] + \mathrm{Re}$ (FiLM)
- **Выход:** $[\bar{u}, \bar{v}, \bar{p}]$

Дополнительно: **стационарный Stokes->Newton** для ламинарного режима $\mathrm{Re} \in [50, 400]$ (fem-генератор `generate_data.py`).

### 3. Стационарное уравнение Дарси (диффузия в пористой среде)

$$-\nabla \cdot (c(x,y)\nabla u(x,y)) = F(x,y), \quad u|_{\partial \Omega}=0$$

- **Вход:** поле проницаемости $c(x,y)$ + описание геометрии (может быть с отверстием)
- **Выход:** поле давления $u(x,y)$
- **Данные:** FEM (MATLAB/Firedrake), CSV на сетке $128 \times 128$

---

## Идея метода

Классический FNO хорошо работает на фиксированном домене с регулярной сеткой, но при смене геометрии качество падает. DNO решает это так:

1. **Диффеоморфизм** $\phi: \Omega_p \to \Omega_u$ --- отображение из физического домена $\Omega_p$ в **универсальный домен** $\Omega_u$ (например, $[0,1]^2$), строится через решение задачи Лапласа на сетке.
2. **Интерполяция** полей (координаты, коэффициент/скорость, решение) на регулярную сетку $128 \times 128$ в $\Omega_u$.
3. **Обучение нейрооператора** в унифицированном представлении.
4. **Применение** к новым геометриям через соответствующее отображение $\phi^{-1}$.

Для доменов с дыркой универсальный домен --- квадрат с фиксированной внутренней окружностью, невалидные точки маскируются (MaskedLpLoss).

---

## Структура репозитория

```
dno/
|-- dno/                          
|   |-- models/
|   |   |-- __init__.py
|   |   \-- dno.py                # DNO: Deep Neural Operator with Geometry Injection
|   |-- data/
|   |   |-- config/
|   |   |   |-- __init__.py
|   |   |   |-- dno_config.py     # DnoDefault, DnoDataConfig, DnoOptConfig
|   |   |   \-- models.py         # DnoModelConfig
|   |   \-- datasets/
|   |       |-- __init__.py
|   |       \-- dno_dataset.py    # DNODataset (darcy, fluid, reservoir cases)
|   |-- layers/
|   |   |-- __init__.py
|   |   \-- embeddings.py        # GridEmbeddingND
|   \-- utils/
|       |-- __init__.py
|       \-- diffeomap/            # Пайплайн диффеоморфизма
|           |-- __init__.py
|           |-- domains.py        # boundary maps (step, square_hole, polygon)
|           |-- harmonic.py       # HarmonicMapper, solve_dirichlet
|           |-- interpolator.py   # GridInterpolator
|           |-- laplacian.py      # Cotangent Laplacian, boundary loops
|           |-- mesh_io.py        # read_msh, read_obj
|           \-- pipeline.py       # run_pipeline (end-to-end)
|-- experiments/
|   |-- scripts/                  #  DNO: Darcy и Reservoir
|   |   |-- diff_train.py         #   Darcy (heptagon with holes)
|   |   |-- obstacle_train.py     #   Navier-Stokes (flow around obstacle)
|   |   \-- reservoir_train.py   #   Reservoir simulation (seq2seq)
|   |-- dns_step/                 #  NS (уступ): DNS + FNO+FiLM
|   |   |-- gen_data_d_step.py    #    Генерация: **нестац. DNS** Firedrake
|   |   |                         #      (Неявный Эйлер, Re∈[800,1500],
|   |   |                         #       осреднение по времени)
|   |   |-- fno/generate_data.py  #    Генерация: **стац. Stokes->Newton**
|   |   |                         #      (ламинарный Re∈[50,400])
|   |   |-- train.py              #   Обучение FNO+FiLM на averaged dataset
|   |   |-- test.py               #   Тестирование
|   |   |-- fno/
|   |   |   |-- model.py          #   FNO2d + FiLM
|   |   |   |-- dataset.py        #   Датасет (backward-facing step)
|   |   |-- create_dif_d_step/    #   Диффеоморфизм для уступа
|   |   \-- dns_averaged_dataset/ #   Предвычисленный датасет
|   |-- quad_test.py              # Тестирование на квадратах
|   |-- test_hole.py              # Тестирование с дыркой
|   |-- train_final.py            # Обучение (без дырок)
|   |-- train_hole_impr.py        # Обучение (с дыркой)
|   |-- pentagon/                 # Пайплайн для пятиугольников
|   |-- seven_w_h/                # Пайплайн "семиугольник с дыркой"
|   |-- sq_art_matlab/            # Пайплайн квадратов (Python + MATLAB)
|   \-- sq_w_hole/                # Пайплайн квадратов с дыркой (Firedrake)
|-- loss/                         # Графики лоссов
|-- pyproject.toml
|-- README.md
|-- LICENSE
\-- .gitignore
```

---

## Установка

```bash
git clone https://github.com/expiwt/DNO.git
cd DNO

# Основные зависимости
pip install -e .
pip install torch numpy zencfg neuralop h5py matplotlib scipy

# Для генерации данных FEM
# pip install firedrake    # Firedrake (Stokes, Darcy с дырками)
# pip install gmsh         # сетки
# MATLAB PDE Toolbox       # квадраты без дырок
```

---

## Быстрый старт

### Использование библиотеки

```python
from dno.models import DNO

model = DNO(n_modes=[16, 16], hidden_channels=32,
            in_channels=4, out_channels=1, n_layers=4)

import torch
x = torch.randn(8, 4, 128, 128)
y = model(x)
print(y.shape)  # (8, 1, 128, 128)
```

### Диффеоморфизм

```python
from dno.utils.diffeomap import run_pipeline

result = run_pipeline(
    mesh_file="geometry.msh",
    mesh_type="gmsh",
    domain_type="polygon",
    resolution=128,
)
print(result.keys())  # x_map, y_map, mask, boundary
```

### Тренировка

```bash
# Darcy (heptagon with holes)
python experiments/scripts/diff_train.py

# Navier-Stokes (obstacle flow)
python experiments/scripts/obstacle_train.py

# Navier-Stokes (backward-facing step, FNO+FiLM)
python experiments/dns_step/train.py

# Reservoir simulation
python experiments/scripts/reservoir_train.py
```

### Генерация данных (Firedrake)

```bash
# Нестационарный DNS (Re ∈ [800, 1500], турбулентный)
python experiments/dns_step/gen_data_d_step.py

# Стационарный Stokes->Newton (Re ∈ [50, 400], ламинарный)
python experiments/dns_step/fno/generate_data.py
```

---

## Поддерживаемые случаи

| case_type   | Уравнение                          | Вход -> Выход                  | Re range      |
|-------------|------------------------------------|-------------------------------|---------------|
| `darcy`     | Стац. Дарси                        | $4 \to 1$ $[C,X,Y,\text{Mask}] \to [U]$ | --- |
| `fluid`     | Стац. Навье–Стокс (обтекание)      | $3 \to 3$ $[X,Y,\mathrm{Re}] \to [U,V,P]$ | --- |
| `reservoir` | Фильтрация (seq2seq)               | $39 \to 72$                   | --- |
| `dns_step`  | **Нестац. DNS** -> осреднение       | $3 \to 3$ $[\text{Mask},\xi,\eta] + \mathrm{Re}$ (FiLM) | $[800, 1500]$ |
| `dns_step` (лам.) | Стац. Stokes->Newton          | то же                         | $[50, 400]$    |

---

## Данные: формат

Во всех вариантах данные приводятся к единому сеточному формату $128 \times 128$.

**Дарси:** NaN внутри дырки -> маска $1/0$ -> отдельный канал + MaskedLpLoss.

**NS (уступ):** физические координаты $[0,L] \times [0,H]$ -> регулярная сетка -> маска жидкости через Point-in-Polygon (лестница уступа).

---

## Лицензия

MIT
