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
    #desempaquetamos la variable shape para tener el # de filas y columnas
    filas, columnas = shape
    #creamos un condicional que se asegura que los valores ingresados sean enteros
    if (filas % 1 != 0) or (columnas % 1 != 0):
        raise ValueError("Los valores ingresados no son números enteros")

    #creamos una matriz vacía que será donde guardemos la matriz que estamos creando
    matriz = []
    #usamos un loop for en el rango de filas y creamos una variable que guardará cada fila de la matriz, llamada mini_matriz
    for fila in range(filas):
        mini_matriz = []
        #usamos un loop for dentro del otro para poder agregar la cantidad de 0 necesaria para cada fila
        for columna in range(columnas):
            mini_matriz.append(0.0)
        #esa cantidad de 0 que se guardó en mini_matriz ahora se agrega como fila de la matriz que vamos a devolver
        matriz.append(mini_matriz)
        #retornamos la matriz
    return matriz



def ones(shape: tuple[int, int]) -> Matriz:

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
    # al igual que en la funcion ceros desempaquetamos shape en filas y columnas
    filas, columnas = shape
    #nos aseguramos que cada variable tenga los datos que dice que nos proporciona
    if (filas % 1 != 0) or (columnas % 1 != 0):
        raise ValueError("Los valores ingresados no son números enteros")
    #creamos la variable matriz que guardará nuestra matriz de unos
    matriz = []
    #usamos el mismo ciclo utilizado en zeros, pero esta vez para unos
    for fila in range(filas):
        mini_matriz = []
        for columna in range(columnas):
            mini_matriz.append(1.0)
        matriz.append(mini_matriz)
    #retornamos la matriz
    return matriz


def identity(n: int) -> Matriz:
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
    #nos aseguramos que n sea un entero y si no retornamos un ValueError
    if n % 1 != 0:
        raise ValueError("Ingrese un número entero")
    #creamos la función matriz que guardará la matriz identidad que estamos creando
    matriz = []
    #usamos un loop for con el rango de n para que sea una matriz nxn
    for fila in range(n):
        #al igual que antes creamos una variable local y temporal llamada mini_matriz que guarda cada fila
        mini_matriz = []
        #lo mismo de antes pero esta vez para las columnas
        for columna in range(n):
            #usamos un condicional que agrega un 1 a mini_matriz si la columna== a fila, si no agrega un cero
            if columna == fila:
                mini_matriz.append(1.0)
            else:
                mini_matriz.append(0.0)
        #de esta forma obtenemos una matriz identidad al agregar cada fila a la matriz
        matriz.append(mini_matriz)
        #retornamos esa matriz
    return matriz


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
    #verificamos que A sea una matriz y no otra cosa
    verificacion = isinstance(A, list)
    #tiramos un ValueError si no es matriz o una lista cuanto menos, ya que al final de cuentas sin numpy una matriz puede ser 
    #list[] o list[list]
    if not verificacion:
        raise ValueError("No se ingresó una matriz")
    #medimos la cantidad de filas y columnas
    filas = len(A)
    columnas = len(A[0])
    return filas, columnas



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
    #verificamos que ambos sean vectores y si no generamos un TypeError
    verificacion_v = isinstance(v, list)
    verificacion_w = isinstance(w, list)
    if not verificacion_v or not verificacion_w:
        raise TypeError("los datos entregados no son vectores")
    #creamos la variable resultado y le guardamos el 0 puesto que todavía no hemos hecho nada

    resultado = 0
    #comparamos que ambos vectoes tengan la misma longitud ya que si no esto no se puede hacer
    if len(v) != len(w):
        raise ValueError("los vectores no tienen la misma longitud")
    
    #calculamos el dot
    for i in range(len(v)):
        resultado += v[i] * w[i]

    #devolvemos el resultado
    return resultado
    


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
    #verificamos que tengan la misma longitud
    if len(v) != len(w):
         raise ValueError("Los vectores deben tener la misma logitud")
    #creamos variable resultado que será un vector
    resultado = []
    #sumamos
    for i in range(len(v)):
        resultado.append(v[i] + w[i])

    #devolvemos resultado
    return resultado
        


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
    #verificamos que sea un vector
    verificacion_v = isinstance(v,list)
    if not verificacion_v:
         raise TypeError
    
    #creamos el vector resultante como una lista
    vector_resultante = []

    #usamos ciclo for para multiplicar por c a todo el vector
    for valor in v:
        resultado = c*valor
        vector_resultante.append(resultado)
    #retornamos valor
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
    #verificamos, a este punto me hubiera venido mejor hacer una función que hiciera esto
    verificar_v = isinstance(v,list)
    if not verificar_v:
        raise TypeError
    
    #hacemos el producto entre el mismo vector que sería lo mismo que hacer el cuadrado
    resultado = dot(v,v)
    #elevamos entre un medio o se puede usar math, como estamos solo con python prefería ver que tanto dura con esto
    return (resultado)**(1/2)


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
    #verificamos que ambos sean matrices
    verificacion_A = isinstance(A,list)
    verificacion_B = isinstance(B,list)
    if not verificacion_A or not verificacion_B:
        raise TypeError
    
    #sacamos sus formas
    fila_A, columna_A = shape(A)
    fila_B,columna_B = shape(B)

    #verificamos que ambas filas y columnas sean iguales
    verificacion_filas = (fila_A == fila_B)
    verificacion_columnas = (columna_A == columna_B)

    #si no da value error
    if not verificacion_filas or not verificacion_columnas:
        raise ValueError
    
    #ahora calculamos la suma de matrices, una vez nos aseguramos que no haya errores

    resultado=[[0 for _ in range(fila_A)]for _ in range(columna_A)]
    for i in range(fila_A):
        for j in range(columna_A):
            resultado[i][j]= A[i][j] + B[i][j]
    
    #devolvemos el resultado
    return resultado



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
    #verificamos
    verificacion_A = isinstance(A,list)
    verificacion_c = isinstance(c,float) or isinstance(c,int)
    if not verificacion_A or not verificacion_c:
        raise TypeError
    
    #obtenemos la forma de la matriz A
    fila,columna = shape(A)

    #creamos una matriz llena de 0
    resultado = [[0 for _ in range(fila)] for _ in range(columna)]

    #reemplazamos esos 0 por los valores que deben ser a traves de un ciclo for
    for i in range(fila):
        for j in range (columna):
            multiplicacion = c * A[i][j]
            resultado[i][j]= multiplicacion
    #devolvemos el valor
    return resultado


    


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
    #verificamos
    verificacion_A = isinstance(A,list)
    verificacion_B = isinstance(B,list)
    if not verificacion_A or not verificacion_B:
        raise TypeError
    
    #obtenemos la forma de A
    
    filas_A,columnas_A= shape(A)

    #nos aseguramos que B sea matriz o vector, si es vector devolvemos el dot para cada fila de A, si es matriz continuamos después
    if all(isinstance(x, (int, float)) for x in B):
        if len(B) != columnas_A:
            raise ValueError
        return [dot(fila,B)for fila in A]
    
    #si es matriz sacamos la forma de B, ya que si lo hubieramos hecho antes nos hubiera dado error
    else:
        filas_B,columnas_B = shape(B)
        #si no son iguales generamos Value Error
        if filas_B != columnas_A:
            raise ValueError
        #trasponemos B ya que es más fácil para calcular el matmul
        B_T = transpose(B)
        #retornamos el dot entre la transpuesta y A, ya que eso nos ayuda a no tener que hacer tantos ciclos
        return [[float(dot(fila_A, columna_B)) for columna_B in B_T] for fila_A in A]



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
    #verificamos
    verificacion_A = isinstance(A,list)
    if not verificacion_A:
        raise TypeError
    
    #vemos si A es solo una 1x1, si es así la devolvemos a ella misma
    if isinstance(A[0],(int,float)):
        #es importante ver que sea cuadrada ya que lo que hace lo de arriba es solo ver si es un vector, eso nos quita algunos errores
        verificacion_cuadrada = len(A)
        if verificacion_cuadrada ==1:
            return A[0]
        else:
            raise ValueError ("La matriz no es cuadrada")
    
    #para los demás casos sacamos la forma de A
    else:
        fila, columna = shape(A)

        #si A no es cuadrada mandamos un ValueError
        if fila != columna:
            raise ValueError ("La matriz no es cuadrada")
        #aquí ponemos cada caso que puede pasar con A
        else:
            #si por si acaso 1x1 no cayó dentro del caso anterior aquí igual lo devolvemos
            if fila == 1:
                return A
            #si es 2x2 usamos la formula de determinantes 2x2 de una
            elif fila == 2:
                determinante = A[0][0]*A[1][1]-A[0][1]*A[1][0]
                return determinante
            
            #para casos 3x3 o mayores se prefirió ir directo a la concatenación, en vez de hacer dos casos separados de 3x3 y de 4x4 o más
            else:
                determinante = 0
                for j in range(fila):
                # Creamos una submatriz que  elimina fila 0 y columna j
                    submatriz = [fila[:j] + fila[j+1:] for fila in A[1:]]
                    #esto es solo para el signo que debe llevar al frente de la fórmula
                    signo = (-1) ** j
                    #aquí hacemos la concatenación donde mangamos a la submatriz a volver a pasar por aquí, así hasta que ya no se puedan hacer más submatrices
                    determinante += signo * A[0][j] * det(submatriz)
                #devolvemos el determinante resuelto
                return determinante

    