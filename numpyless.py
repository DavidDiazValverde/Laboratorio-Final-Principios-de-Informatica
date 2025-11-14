"""
╔════════════════════════════════════════════════════════════════════════╗
║                            📐 NumpyLess                                ║
║                  Pure Python Linear Algebra Library                    ║
║                      (NumPy-less, stress-more!)                        ║
╚════════════════════════════════════════════════════════════════════════╝

Una biblioteca minimalista de álgebra lineal que implementa operaciones
tipo NumPy usando solo Python puro. ¡Perfecta para entender qué pasa
"bajo el capó"!

Uso Recomendado:
    import numpyless as npl

    # O para máxima ironía:
    import numpyless as np  # ¡Cuidado con esto! 😈

Tipos de Datos:
- Vector: list[float] - Un array 1D de flotantes
- Matriz: list[list[float]] - Un array 2D de flotantes (filas x columnas)
"""

# --- Alias de Tipos Nativos ---
Vector = list[float]
Matriz = list[list[float]]

# -------------------------------------------------------------------
# Sección 1: Creación de Arrays (⭐ Básico)
# -------------------------------------------------------------------


def zeros(shape: tuple[int, int]) -> Matriz:
    try:
        filas, columnas = shape
        if (filas % 1 != 0) or (columnas % 1 != 0):
            raise ValueError("Los valores ingresados no son números enteros")
        matriz = []
        for fila in range(filas):
            mini_matriz = []
            for columna in range(columnas):
                mini_matriz.append(0.0)
            matriz.append(mini_matriz)
        return matriz
    except ValueError as e:
        print(f"!Error {e}")
    """Crea una matriz rellena de ceros.

    Equivalente en NumPy: np.zeros(shape)

    Args:
        shape: Tupla (filas, columnas) que define las dimensiones.

    Returns:
        Matriz: Una matriz de shape con valores 0.0.

    Ejemplo:
        >>> zeros((2, 3))
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    Pista: Usa listas por comprensión anidadas
    """


def ones(shape: tuple[int, int]) -> Matriz:
    try:
        filas, columnas = shape
        if (filas % 1 != 0) or (columnas % 1 != 0):
            raise ValueError("Los valores ingresados no son números enteros")
        matriz = []
        for fila in range(filas):
            mini_matriz = []
            for columna in range(columnas):
                mini_matriz.append(1.0)
            matriz.append(mini_matriz)
        return matriz
    except ValueError as e:
        print(f"!Error {e}")

    """Crea una matriz rellena de unos.

    Equivalente en NumPy: np.ones(shape)

    Args:
        shape: Tupla (filas, columnas) que define las dimensiones.

    Returns:
        Matriz: Una matriz de shape con valores 1.0.

    Ejemplo:
        >>> ones((2, 2))
        [[1.0, 1.0], [1.0, 1.0]]

    Pista: Similar a zeros() pero con 1.0
    """


def identity(n: int) -> Matriz:
    try:
        if n % 1 != 0:
            raise ValueError("Ingrese un número entero")
        matriz = []
        for fila in range(n):
            mini_matriz = []
            for columna in range(n):
                if columna == fila:
                    mini_matriz.append(1.0)
                else:
                    mini_matriz.append(0.0)
            matriz.append(mini_matriz)
        return matriz
    except ValueError as e:
        print(f"Error!! {e}")

    """Crea una matriz identidad cuadrada.

    Equivalente en NumPy: np.identity(n)

    Args:
        n: El tamaño (número de filas y columnas) de la matriz.

    Returns:
        Matriz: Una matriz identidad de n x n.

    Ejemplo:
        >>> identity(3)
        [[1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0]]

    Pista: La diagonal tiene 1.0 cuando fila == columna
    """


# -------------------------------------------------------------------
# Sección 2: Información de Arrays (⭐ Básico)
# -------------------------------------------------------------------


def shape(A: Matriz) -> tuple[int, int]:
    """Devuelve las dimensiones de una matriz como (filas, columnas).

    Equivalente en NumPy: A.shape

    Args:
        A: La matriz de entrada.

    Returns:
        tuple[int, int]: Una tupla (filas, columnas).

    Ejemplo:
        >>> shape([[1, 2, 3], [4, 5, 6]])
        (2, 3)

    Pista: len(A) da filas, len(A[0]) da columnas
    """
    try:
        verificacion = isinstance(A, list)
        if not verificacion:
            raise ValueError("No se ingresó una matriz")
        filas = len(A)
        columnas = len(A[0])
        return filas, columnas
    except ValueError as e:
        print(f"Error {e}")


def transpose(A: Matriz) -> Matriz:
    """Devuelve la transpuesta de una matriz A.

    La transpuesta intercambia filas por columnas: A_t[j][i] = A[i][j].

    Equivalente en NumPy: A.T o np.transpose(A)

    Args:
        A: La matriz de entrada.

    Returns:
        Matriz: La matriz transpuesta.

    Ejemplo:
        >>> transpose([[1, 2, 3], [4, 5, 6]])
        [[1, 4], [2, 5], [3, 6]]

    Pista: Usa zip(*A) o listas por comprensión
    """
    try:
        verificacion = isinstance(A, list)
        if not verificacion:
            raise ValueError("No se ingresó una matriz")
        filas = len(A)
        columnas = len(A[0])
        transpuesta = [[0 for _ in range(filas)] for _ in range(columnas)]

        for fila in range(filas):
            for columna in range(columnas):
                transpuesta[columna][fila] = A[fila][columna]
        return transpuesta
    except ValueError as e:
        print(f"Error {e}")


# -------------------------------------------------------------------
# Sección 3: Operaciones con Vectores (⭐⭐ Intermedio)
# -------------------------------------------------------------------


def dot(v: Vector, w: Vector) -> float:
    """Calcula el producto punto (producto escalar) de dos vectores.

    Fórmula: v · w = v[0]*w[0] + v[1]*w[1] + ... + v[n]*w[n]

    Equivalente en NumPy: np.dot(v, w)

    Args:
        v: El primer vector.
        w: El segundo vector.

    Returns:
        float: El resultado del producto punto.

    Raises:
        ValueError: Si los vectores no tienen la misma dimensión.

    Ejemplo:
        >>> dot([1, 2, 3], [4, 5, 6])
        32.0  # = 1*4 + 2*5 + 3*6

    Pista: Usa sum() y zip()
    """
    try:
        verificacion_v = isinstance(v, list)
        verificacion_w = isinstance(w, list)
        if not verificacion_v or not verificacion_w:
            raise TypeError("los datos entregados no son vectores")

        resultado = 0
        if len(v) != len(w):
            raise ValueError("los vectores no tienen la misma longitud")
        for i in range(len(v)):
            resultado += v[i] * w[i]
        return resultado
    except ValueError:
        raise ValueError
    except TypeError:
        raise TypeError


def add(v: Vector, w: Vector) -> Vector:
    """Suma dos vectores elemento a elemento.

    Equivalente en NumPy: v + w

    Args:
        v: El primer vector.
        w: El segundo vector.

    Returns:
        Vector: El vector resultante de la suma.

    Raises:
        ValueError: Si los vectores no tienen la misma dimensión.

    Ejemplo:
        >>> add([1, 2], [3, 4])
        [4.0, 6.0]

    Pista: Usa listas por comprensión con zip()
    """
    try:
        if len(v) != len(w):
            raise ValueError("Los vectores deben tener la misma logitud")
        resultado = []
        for i in range(len(v)):
            resultado.append(v[i] + w[i])

        return resultado
    except ValueError:
        raise ValueError
        


def multiply(c: float, v: Vector) -> Vector:
    """Multiplica cada elemento de un vector por un escalar.

    Equivalente en NumPy: c * v

    Args:
        c: El escalar.
        v: El vector.

    Returns:
        Vector: El vector resultante escalado.

    Ejemplo:
        >>> multiply(2.5, [1, 2, 3])
        [2.5, 5.0, 7.5]

    Pista: Multiplica c por cada elemento
    """

    verificacion_v = isinstance(v,list)
    if not verificacion_v:
         raise TypeError
    
    vector_resultante = []

    for valor in v:
        resultado = c*valor
        vector_resultante.append(resultado)
    return vector_resultante
        
        


def norm(v: Vector) -> float:
    """Calcula la magnitud (norma L2) de un vector.

    Fórmula: ||v|| = sqrt(v[0]² + v[1]² + ... + v[n]²)

    Equivalente en NumPy: np.linalg.norm(v)

    Args:
        v: El vector.

    Returns:
        float: La magnitud del vector.

    Ejemplo:
        >>> norm([3, 4])
        5.0  # = sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25)

    Pista: Usa dot(v, v) y luego sqrt() del módulo math
    """
    raise NotImplementedError("Función no implementada.")


# -------------------------------------------------------------------
# Sección 4: Operaciones con Matrices (⭐⭐ Intermedio)
# -------------------------------------------------------------------


def add_matrices(A: Matriz, B: Matriz) -> Matriz:
    """Suma dos matrices elemento a elemento.

    Equivalente en NumPy: A + B

    Args:
        A: La primera matriz.
        B: La segunda matriz.

    Returns:
        Matriz: La matriz resultante de la suma.

    Raises:
        ValueError: Si las matrices no tienen la misma forma.

    Ejemplo:
        >>> add_matrices([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        [[6.0, 8.0], [10.0, 12.0]]

    Pista: Suma elemento a elemento, fila por fila
    """
    raise NotImplementedError("Función no implementada.")


def multiply_matrix(c: float, A: Matriz) -> Matriz:
    """Multiplica cada elemento de una matriz por un escalar.

    Equivalente en NumPy: c * A

    Args:
        c: El escalar.
        A: La matriz.

    Returns:
        Matriz: La matriz resultante escalada.

    Ejemplo:
        >>> multiply_matrix(2, [[1, 2], [3, 4]])
        [[2.0, 4.0], [6.0, 8.0]]

    Pista: Similar a multiply() pero para cada fila
    """
    raise NotImplementedError("Función no implementada.")


def matmul(A: Matriz, B: Matriz | Vector) -> Matriz | Vector:
    """Multiplica una matriz A por una matriz B o vector v.

    Regla: El número de columnas de A debe ser igual al número de
           filas de B (o longitud de v).

    Equivalente en NumPy: A @ B

    Args:
        A: La matriz izquierda (m × n).
        B: La matriz derecha (n × p) o vector (n).

    Returns:
        Matriz (m × p) o Vector (m): El resultado de la multiplicación.

    Raises:
        ValueError: Si las dimensiones no son compatibles.

    Ejemplos:
        >>> matmul([[1, 2]], [3, 4])
        [11.0]  # = [1*3 + 2*4]

        >>> matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        [[19.0, 22.0], [43.0, 50.0]]

    Pista: Para matrices, cada elemento resultado[i][j] es el
           producto punto de la fila i de A con la columna j de B
    """
    raise NotImplementedError("Función no implementada.")


# -------------------------------------------------------------------
# Sección 5: Álgebra Lineal (⭐⭐⭐ Avanzado - Opcional/Extra)
# -------------------------------------------------------------------


def det(A: Matriz) -> float:
    """Calcula el determinante de una matriz cuadrada.

    NOTA: Esta es la función más difícil. Es opcional pero da puntos extra.

    Para matriz 2×2:
        det([[a, b], [c, d]]) = a*d - b*c

    Para matriz 3×3 y mayores:
        Usa expansión de cofactores (recursivo) o eliminación gaussiana.

    Equivalente en NumPy: np.linalg.det(A)

    Args:
        A: La matriz cuadrada.

    Returns:
        float: El valor del determinante.

    Raises:
        ValueError: Si la matriz no es cuadrada.

    Ejemplos:
        >>> det([[4, 3], [2, 1]])
        -2.0  # = 4*1 - 3*2

        >>> det([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        1.0  # determinante de identidad = 1

    Pistas:
    - Caso base: matriz 1×1 devuelve el único elemento
    - Caso 2×2: usa la fórmula directa
    - Caso 3×3+: expansión por primera fila (recursivo)
    """
    raise NotImplementedError("Función no implementada.")
