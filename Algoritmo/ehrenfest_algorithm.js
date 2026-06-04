/**
 * ============================================================
 *  CADENA DE EHRENFEST — Algoritmo completo
 *  Procesos Estocásticos UD 2026-1
 * ============================================================
 *
 *  Modelo: N bolas numeradas distribuidas entre dos urnas A y B.
 *  En cada paso se elige una bola al azar (uniforme en {1,...,N})
 *  y se transfiere a la otra urna.
 *
 *  Estado: Xn = numero de bolas en urna A
 *  Espacio de estados: S = {0, 1, ..., N}
 *
 *  Probabilidades de transicion:
 *    p(i, i-1) = i/N        (mover una bola de A -> B)
 *    p(i, i+1) = (N-i)/N    (mover una bola de B -> A)
 *    p(0, 1)   = 1          (estado reflector inferior)
 *    p(N, N-1) = 1          (estado reflector superior)
 * ============================================================
 */


// ============================================================
//  UTILIDADES MATEMATICAS
// ============================================================

/**
 * Calcula el coeficiente binomial C(n, k) = n! / (k! * (n-k)!)
 * usando multiplicacion incremental para evitar desbordamiento.
 * @param {number} n
 * @param {number} k
 * @returns {number}
 */
function binom(n, k) {
  if (k < 0 || k > n) return 0;
  if (k === 0 || k === n) return 1;
  let result = 1;
  for (let i = 0; i < k; i++) {
    result = result * (n - i) / (i + 1);
  }
  return result;
}

/**
 * Multiplica dos matrices cuadradas A y B de tamaño n*n.
 * @param {number[][]} A
 * @param {number[][]} B
 * @returns {number[][]} producto A*B
 */
function matMul(A, B) {
  const n = A.length;
  const C = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++)
    for (let k = 0; k < n; k++)
      if (A[i][k] !== 0)
        for (let j = 0; j < n; j++)
          C[i][j] += A[i][k] * B[k][j];
  return C;
}

/**
 * Eleva una matriz cuadrada M a la potencia entera exp
 * mediante exponenciacion rapida: O(n^3 * log(exp)) operaciones.
 * @param {number[][]} M
 * @param {number} exp  - exponente no negativo
 * @returns {number[][]} M^exp
 */
function matPow(M, exp) {
  const size = M.length;
  // Inicializar resultado como la identidad I
  let result = Array.from({ length: size }, (_, i) =>
    Array.from({ length: size }, (_, j) => (i === j ? 1 : 0))
  );
  let base = M.map(row => row.slice()); // copia de M para no mutarla

  while (exp > 0) {
    if (exp & 1) result = matMul(result, base); // bit menos significativo activo
    base = matMul(base, base);                   // cuadrar la base
    exp >>= 1;                                   // desplazar bits a la derecha
  }
  return result;
}

/**
 * Resuelve el sistema lineal cuadrado A*x = b mediante
 * eliminacion gaussiana con pivoteo parcial.
 * @param {number[][]} A  - matriz de coeficientes (se modifica in-place en copia)
 * @param {number[]}   b  - vector de terminos independientes
 * @returns {number[]} solucion x, o vector de NaN si el sistema es singular
 */
function gaussianElimination(A, b) {
  const n = A.length;
  // Construir matriz aumentada [A | b]
  const aug = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    // Buscar el pivote con mayor valor absoluto en la columna actual
    let pivot = col;
    for (let r = col + 1; r < n; r++)
      if (Math.abs(aug[r][col]) > Math.abs(aug[pivot][col])) pivot = r;

    // Intercambiar fila actual con la del pivote
    [aug[col], aug[pivot]] = [aug[pivot], aug[col]];

    const pv = aug[col][col];
    if (Math.abs(pv) < 1e-12) continue; // columna casi nula, continuar

    // Normalizar la fila pivote
    for (let c = col; c <= n; c++) aug[col][c] /= pv;

    // Eliminar el coeficiente de esta columna en todas las demas filas
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = aug[r][col];
      for (let c = col; c <= n; c++) aug[r][c] -= factor * aug[col][c];
    }
  }

  // Extraer solucion de la columna de terminos independientes
  return aug.map(row => row[n]);
}


// ============================================================
//  1. MATRIZ DE TRANSICION
// ============================================================

/**
 * Construye la matriz de transicion P de tamaño (N+1)*(N+1)
 * para la cadena de Ehrenfest con N bolas.
 *
 * Estructura tridiagonal:
 *   P[i][i-1] = i/N       para i = 1, ..., N
 *   P[i][i+1] = (N-i)/N   para i = 0, ..., N-1
 *   todos los demas = 0
 *
 * @param {number} N - numero total de bolas
 * @returns {number[][]} matriz P de tamaño (N+1)*(N+1)
 */
function buildTransitionMatrix(N) {
  const size = N + 1;
  const P = Array.from({ length: size }, () => new Array(size).fill(0));

  for (let i = 0; i <= N; i++) {
    if (i > 0) P[i][i - 1] = i / N;        // prob. de bajar: mover bola A -> B
    if (i < N) P[i][i + 1] = (N - i) / N;  // prob. de subir: mover bola B -> A
  }
  return P;
}


// ============================================================
//  2. CLASES DE COMUNICACION
// ============================================================

/**
 * Determina las clases de comunicacion de la cadena de Ehrenfest.
 *
 * La cadena es IRREDUCIBLE: todos los estados se comunican entre si,
 * por lo que existe una UNICA clase de comunicacion C = {0, 1, ..., N}.
 *
 * Desde cualquier estado i se puede alcanzar cualquier estado j en
 * a lo sumo N pasos (ya que la cadena se mueve ±1 en cada paso).
 *
 * Nota: la cadena tiene periodo d = 2 (solo se regresa al mismo estado
 * en un numero par de pasos), pero esto NO implica clases separadas.
 * El periodo es una propiedad de los estados dentro de la unica clase.
 *
 * @param {number} N
 * @returns {{ states: number[], irreducible: boolean, period: number }}
 */
function getCommunicationClasses(N) {
  const states = Array.from({ length: N + 1 }, (_, i) => i);
  return {
    states,            // unica clase de comunicacion: {0, 1, ..., N}
    irreducible: true, // todos los estados se comunican
    period: 2          // periodo comun de la clase
  };
}


// ============================================================
//  3. CLASIFICACION DE ESTADOS
// ============================================================

/**
 * Clasifica cada estado de la cadena de Ehrenfest.
 *
 * Resultado teorico:
 *   - TODOS los estados son RECURRENTES POSITIVOS.
 *   - No existen estados transitorios ni recurrentes nulos.
 *   - Todos tienen periodo d = 2.
 *
 * Justificacion: la cadena es irreducible sobre un espacio finito,
 * luego todos sus estados son recurrentes. Como pi(i) > 0 para todo i,
 * todos son recurrentes positivos con E[Ti] = 1/pi(i) < infinito.
 *
 * @param {number} N
 * @returns {Array<{ state: number, type: string, period: number }>}
 */
function classifyStates(N) {
  return Array.from({ length: N + 1 }, (_, i) => ({
    state: i,
    type: 'recurrente positivo', // ningun estado es transitorio
    period: 2                    // periodo 2: solo se regresa en pasos pares
  }));
}


// ============================================================
//  4. DISTRIBUCION ESTACIONARIA
// ============================================================

/**
 * Calcula la distribucion estacionaria pi de la cadena de Ehrenfest.
 *
 * Resultado exacto: es la distribucion Binomial(N, 1/2):
 *
 *   pi(i) = C(N, i) / 2^N    para i = 0, 1, ..., N
 *
 * Verificacion por balance detallado (reversibilidad):
 *   pi(i) * p(i, i+1) = pi(i+1) * p(i+1, i)
 *   [C(N,i)/2^N] * [(N-i)/N] = [C(N,i+1)/2^N] * [(i+1)/N]  OK
 *
 * @param {number} N
 * @returns {number[]} vector pi de longitud N+1, con sum(pi) = 1
 */
function stationaryDistribution(N) {
  const twoN = Math.pow(2, N);
  return Array.from({ length: N + 1 }, (_, i) => binom(N, i) / twoN);
}


// ============================================================
//  5. DISTRIBUCION DE Xn DESPUES DE n PASOS
// ============================================================

/**
 * Calcula la distribucion de Xn dado que X0 = i0,
 * multiplicando el vector de distribucion inicial por P iterativamente.
 *
 *   P(Xn = j | X0 = i0) = (e_{i0} * P^n)[j]
 *
 * Metodo iterativo: util para n pequeño (O(n * N^2)).
 *
 * @param {number[][]} P   - matriz de transicion
 * @param {number} i0      - estado inicial
 * @param {number} steps   - numero de pasos n
 * @returns {number[]} distribucion de Xn (vector de longitud N+1)
 */
function distributionAfterSteps(P, i0, steps) {
  const N = P.length - 1;

  // Vector inicial: toda la masa probabilistica concentrada en i0
  let dist = new Array(N + 1).fill(0);
  dist[i0] = 1;

  // En cada paso: dist_nueva[j] = sum_i dist[i] * P[i][j]
  for (let s = 0; s < steps; s++) {
    const next = new Array(N + 1).fill(0);
    for (let i = 0; i <= N; i++)
      if (dist[i] > 0)
        for (let j = 0; j <= N; j++)
          next[j] += dist[i] * P[i][j];
    dist = next;
  }
  return dist;
}


// ============================================================
//  7. TIEMPOS MEDIOS DE RETORNO (RECURRENCIA)
// ============================================================

/**
 * Calcula el tiempo medio de primer retorno al estado i:
 *
 *   E[Ti] = 1 / pi(i) = 2^N / C(N, i)
 *
 * Resultado del teorema ergodico: para una cadena irreducible
 * y recurrente positiva, E[Ti] = 1/pi(i).
 *
 * @param {number} N    - numero total de bolas
 * @param {number[]} pi - distribucion estacionaria
 * @returns {number[]} vector de tiempos medios de retorno (longitud N+1)
 */
function meanReturnTimes(N, pi) {
  return pi.map(p => 1 / p); // E[Ti] = 1/pi(i)
}


// ============================================================
//  8. TIEMPOS MEDIOS DE PRIMERA VISITA (HITTING TIMES)
// ============================================================

/**
 * Calcula el tiempo medio de primera visita desde src hasta dst:
 *   E[T_{src -> dst}]
 *
 * Se resuelve el sistema de ecuaciones de primer paso:
 *
 *   h(dst) = 0
 *   h(i)   = 1 + sum_j P(i,j) * h(j)   para i != dst
 *
 * Reescrito como (I - P_d) * h = 1  donde P_d es P con la fila
 * y columna de dst puestas a cero. Se excluye la ecuacion de dst
 * y se llena h(dst)=0 directamente.
 *
 * @param {number[][]} P  - matriz de transicion
 * @param {number} src    - estado de partida
 * @param {number} dst    - estado objetivo
 * @returns {number} E[T_{src -> dst}] en numero de pasos esperados
 */
function meanHittingTime(P, src, dst) {
  const N = P.length - 1;
  const size = N + 1;

  // Caso trivial: origen = destino -> es el tiempo de retorno
  if (src === dst) return 1 / stationaryDistribution(N)[src];

  // Lista de estados activos (todos menos dst)
  const active = [];
  for (let i = 0; i <= N; i++) if (i !== dst) active.push(i);
  const m = active.length; // dimension del sistema reducido

  // Construir la matriz (I - P_d) restringida a estados activos,
  // y el vector de terminos independientes b = 1
  const A = Array.from({ length: m }, () => new Array(m).fill(0));
  const b = new Array(m).fill(1);

  for (let r = 0; r < m; r++) {
    const i = active[r];
    for (let c = 0; c < m; c++) {
      const j = active[c];
      A[r][c] = (i === j ? 1 : 0) - P[i][j]; // (I - P)[i][j] sobre estados activos
    }
    // La contribucion de P[i][dst] va al lado derecho como termino conocido:
    //   h(i) = 1 + P[i][dst]*0 + sum_{j!=dst} P[i][j]*h(j)
    // -> b[r] = 1 (la contribucion de dst ya es 0 porque h(dst)=0)
  }

  // Resolver el sistema A * h_active = b
  const hActive = gaussianElimination(A, b);

  // Reconstruir vector completo de hitting times
  const h = new Array(size).fill(0); // h(dst) = 0
  for (let r = 0; r < m; r++) h[active[r]] = hActive[r];

  return h[src];
}

/**
 * Calcula todos los tiempos medios de primera visita desde src
 * hacia cada estado j en {0,...,N}.
 *
 * @param {number[][]} P  - matriz de transicion
 * @param {number} src    - estado de partida
 * @returns {number[]} vector h donde h[j] = E[T_{src->j}]
 */
function allHittingTimesFrom(P, src) {
  const N = P.length - 1;
  return Array.from({ length: N + 1 }, (_, j) => meanHittingTime(P, src, j));
}
