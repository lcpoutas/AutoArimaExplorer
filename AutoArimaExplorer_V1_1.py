from __future__ import annotations
import math
from typing import Dict, Tuple, Optional, Any, Iterable, List
import numpy as np
import pandas as pd
from pydantic import BaseModel, conint
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,  # o DEBUG en desarrollo
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

# --------- Contenedor de resultados (pydantic para validación) ----------

class ModelSpec(BaseModel):
    p: conint(ge=0)
    d: conint(ge=0)
    q: conint(ge=0)
    res: Optional[Any] = None # ARIMAResults (puede venir converged=False)

class AutoArimaExplorer:
    """
    Exploración de ARIMA/ARMA (p,d,q) con:
        - d = 0: ARMA sobre log-retornos (estacionarios)
        - d >= 1: ARIMA con diferenciación interna sobre niveles estandarizados
        - Reintentos de ajuste (statespace + method_kwargs; innovations_mle en d=0)
        - Selección por BIC con Ljung–Box
    """

    def __init__(self, y: pd.Series, name: str = 'target', logger: Optional[logging.Logger] = None):
        self.name = name
        self.y_raw = pd.to_numeric(y, errors='coerce').dropna().astype(float)
        self.logger = logger or logging.getLogger(__name__)
        
        # Serie de niveles estandarizada (para d>=1)
        y_lvls = self.y_raw.copy()
        muL, sdL = y_lvls.mean(), y_lvls.std()
        if sdL == 0:
            raise ValueError("La serie de niveles es (casi) constante; ARIMA no es identificable.")
        self.y_lvls = (y_lvls - muL) / sdL

        # ------ Serie de log-retornos (para d=0) -------------
        
        # 1) Valor mínimo de la serie en niveles para comprobar si hay valores <= 0
        min_v = float(self.y_raw.min())
        # 2) Aseguramos positividad antes del log
        y_pos = self.y_raw + (abs(min_v) + 1.0) if min_v <=0 else self.y_raw
        # 3) Transformación logarítmica -> estabiliza varianza y convierte cambios multiplicativos en aditivos
        y_log = np.log(y_pos)
        # 4) Calculamos log-retornos: r_t = log(y_t) − log(y_{t-1}) y aplicamos limpieza con replace
        y_ret = y_log.diff().replace([np.inf, -np.inf], np.nan).dropna()
        # 6) Estandarización opcional (z-score):
        muR, sdR = y_ret.mean(), y_ret.std()
        self.y_ret = (y_ret - muR)/sdR if (sdR > 0 and np.isfinite(sdR)) else y_ret
        
        # Almacenes
        self.results: Dict[Tuple[int, int, int], ModelSpec] = {}
        self.errors: Dict[Tuple[int, int, int], str] = {}

    # ------------- Utilidades Internas ---------------

    def _series_for_d(self, d:int) -> pd.Series:
        return self.y_ret if d == 0 else self.y_lvls

    def _fit_candidate_trials(self,
                             y: pd.Series,
                             order: Tuple[int, int, int],
                            ) -> Tuple[Optional[Any], Optional[dict], Optional[str]]:
        """
            Ajusta un candidato ARIMA(p, d, q) sobre la, serie 'y' probando varios *trials*
            (combinaciones de trend / restricciones / estimator) y devuelve el mejor por BIC.

            Returns
            ---------------------------------------
            (best_res, best_cfg, err_msg)
              - best_res : ARIMAResultsWrapper o None si todos fallan
              - best_cfg : dict con la config ganadora (o None)
              - err_msg  : último error capturado si no hubo éxito
        """

        p, d, q = order

        # --- constrir lista de intentos (trials) según d ---
        trials: List[dict] = []
        if d == 0:
            # Arma sobre retornos: probar sin constante ('n') y con constante ('c')
            for tr in ('n', 'c'):
                trials.append(dict( #1) innovations_mle (rapido/estable)
                    estimator='innovations_mle', trend=tr,
                    enforce_stationarity=True, enforce_invertibility=True
                ))
                trials.append(dict(  # 2) statespace + L-BFGS (fallback)
                    estimator='statespace', trend=tr,
                    enforce_stationarity=True, enforce_invertibility=True,
                    ss_kwargs={'method': 'lbfgs', 'maxiter': 3000, 'disp': 0}
                ))
        elif d==1:
            # ARIMA con una diferencia: drift ('t') o sin drift ('n')
            for tr in ('t', 'n'):
                trials.append(dict( # sin restricciones (permite raices ~1)
                    estimator='statespace', trend=tr,
                    enforce_stationarity=False, enforce_invertibility=False,
                    ss_kwargs={'method': 'lbfgs', 'maxiter': 3000, 'disp': 0}
                ))
                trials.append(dict(  # con restricciones (más estable)
                    estimator='statespace', trend=tr,
                    enforce_stationarity=True, enforce_invertibility=True,
                    ss_kwargs={'method': 'lbfgs', 'maxiter': 3000, 'disp': 0}
                ))
        else:
            # d >= 2: por teoría, 'n' (tendencias de orden < d se diferencian fuera)
            tr = 'n'
            trials.append(dict(
                estimator='statespace', trend=tr,
                enforce_stationarity=False, enforce_invertibility=False,
                ss_kwargs={'method': 'lbfgs', 'maxiter': 3000, 'disp': 0}
            ))
            trials.append(dict(
                estimator='statespace', trend=tr,
                enforce_stationarity=True, enforce_invertibility=True,
                ss_kwargs={'method': 'lbfgs', 'maxiter': 4000, 'disp': 0}
            ))

        best_res, best_cfg = None, None
        last_err = None

        # --- ejecutar trials y quedarnos con el mejor por bic
        for cfg in trials:
            try:
                m = ARIMA(
                    y, order=order, trend=cfg['trend'],
                    enforce_stationarity=cfg['enforce_stationarity'],
                    enforce_invertibility= cfg['enforce_invertibility']
                )
                if cfg['estimator'] == 'innovations_mle':
                    res = m.fit(method='innovations_mle') # no admite method_kwargs
                else:
                    res = m.fit(method='statespace', method_kwargs=cfg['ss_kwargs'])
                    
                # BIC finito como condición mínima; descartar resultados patológicos
                # (Suelen indicar modelo mal especificado o mal condicionado)
                bic = getattr(res, 'bic', float('inf')) # si no hay BIC -> +inf
                if not math.isfinite(bic):
                    continue

                # Si es el primero o mejora al mejor hasta ahora -> actualizar ganador
                if (best_res is None) or (res.bic < best_res.bic):
                    best_res, best_cfg = res, cfg

            except Exception as e:
                last_err = str(e)
                continue

        return best_res, best_cfg, last_err

    def _fit_candidate_insample(
        self,
        order: Tuple[int, int, int],
    ) -> Tuple[Optional[Any], Optional[str]]:
        """
        Ajusta un ARIMA(p,d,q) **in-sample** usando la serie interna apropiada a `d`.
    
        Parámetros
        ----------
        order : (p, d, q)
            Especificación del ARIMA a ajustar.
    
        Comportamiento
        --------------
        - Si d == 0  → usa self.y_ret (log-retornos).
        - Si d >= 1 → usa self.y_lvls (niveles estandarizados).
        - Ejecuta la política de reintentos definida en `_fit_candidate_trials`
          (innovations_mle / statespace con distintas restricciones) y
          **elige el mejor intento por BIC**.
    
        Returns
        -------
        (res, err) :
            - res : ARIMAResultsWrapper si al menos un intento fue exitoso; None en caso contrario.
            - err : str con el último error capturado si todos los intentos fallan; None si hubo éxito.
        """
        _, d, _ = order
    
        # Selección de la serie según d (usa el helper si existe)
        y_use = self._series_for_d(d) 
    
        # Serie vacía → no intentamos ajustar
        if y_use is None or len(y_use) == 0:
            return None, "empty series for d"
    
        # Ajuste con reintentos y selección por BIC
        res, _, err = self._fit_candidate_trials(y_use, order)
        return res, err


    def _fit_candidate_on_subset(
        self,
        y_sub: pd.Series,
        order: Tuple[int, int, int],
    ) -> Tuple[Optional[Any], Optional[str]]:
        """
        Ajusta un ARIMA(p,d,q) **sobre una subserie concreta** (útil para validación
        rolling / ventanas OOS), reutilizando la misma política de reintentos y
        selección por BIC que in-sample.
    
        Parámetros
        ----------
        y_sub : pd.Series
            Subserie de trabajo (por ejemplo, una ventana temporal específica).
            Debe estar preprocesada por el llamador (limpieza/transformaciones).
        order : (p, d, q)
            Especificación del ARIMA a ajustar en la subserie.
    
        Returns
        -------
        (res, err) :
            - res : ARIMAResultsWrapper si algún intento tuvo éxito; None si todos fallan.
            - err : str con el último error capturado si no hubo éxito; None si ajustó bien.
        """
        # Subserie vacía → no intentamos ajustar
        if y_sub is None or len(y_sub) == 0:
            return None, "empty subset series"
    
        # Ajuste con reintentos y selección por BIC en la subserie
        res, _, err = self._fit_candidate_trials(y_sub, order)
        return res, err

    @staticmethod
    def _ljung_box_ok(res, lags=(10, 20, 40), alpha=0.05, check_sq: bool = False) -> bool:
        """
        Guardarraíl Ljung–Box:
          - Comprueba ausencia de autocorrelación en residuos (media).
          - Si `check_sq=True`, también lo hace en residuos^2 (varianza).
        Devuelve True si TODOS los p-valores en los lags considerados son ≥ alpha.
        """
        try:
            r = np.asarray(res.resid, dtype=float).ravel()
            r = r[np.isfinite(r)]
            n = r.size
            if n < 8:
                return True  # no penalizamos con muestras muy cortas
    
            # Lags adaptativos y seguros
            max_auto = max(5, min(40, int(np.sqrt(n))))  # típico: sqrt(n) acotado
            lags_use = [h for h in lags if 1 <= h <= max_auto]
            if not lags_use:  # si todos eran demasiado grandes, usa max_auto
                lags_use = [max_auto]
    
            # Ljung–Box en residuos
            lb = acorr_ljungbox(r, lags=lags_use, return_df=True)
            ok_mean = (lb["lb_pvalue"] >= alpha).all()
    
            if not check_sq:
                return bool(ok_mean)
    
            # (opcional) Ljung–Box en residuos^2
            lb2 = acorr_ljungbox(r * r, lags=lags_use, return_df=True)
            ok_var = (lb2["lb_pvalue"] >= alpha).all()
    
            return bool(ok_mean and ok_var)

        except Exception as e:
            logger.debug("LB guardrail fallback (excepción: %s)", e)
            return True  # fallback conservador

    @staticmethod
    def _has_enough_data(n: int, p: int, d: int, q: int, base: int = 50, per_param: int = 10) -> bool:
        """
            Heurística Box–Jenkins:
              - suelo absoluto: base (30–50 típico)
              - múltiplo de complejidad: per_param * (p + q + d + 1)
            Devuelve True si hay datos suficientes para estimar ARIMA(p,d,q).
        """
            
        return (n - d) >= max(base, per_param * (p + q + d + 1))

    @staticmethod
    def _winkler_score(y_true: float, lower: float, upper: float, alpha: float = 0.05) -> float:
        """
            Winkler score para un (1-alpha) PI: ancho + penalización si el valor cae fuera.
            Menor es mejor
        """

        w = float(upper - lower)
        if y_true < lower:
            w += 2.0 / alpha * (lower - y_true)
        elif y_true > upper:
            w += 2.0 / alpha * (y_true - upper)
        return w

    def _result_ok(self,
                   res: Any,
                   *,
                   require_converged: bool = True,
                   require_finite_bic: bool = True,
                   check_lb: bool = False,
                   lb_lags = (10, 20, 40, 60),
                   alpha: float = 0.05,
                   check_sq: bool = False,
                  ) -> bool:
        """
            Filtro rápido de calidad del resultado de ajuste

            - require_converged: si True, descarta modelos con convergencia reportada como False.
            - require_finite_bic: si True, exige BIC finito (evita soluciones patológicas).
            - check_lb: si True, aplica guardarraíl Ljung–Box (media y opcionalmente varianza con check_sq).
            Devuelve True si el resultado pasa todos los filtros.        
        """

        # 1) Convergencia (cuando el backend la reporta
        if require_converged and getattr(res, 'converged', True) is False:
            return False

        # 2) Bic finito (evita NaN/inf
        if require_finite_bic:
            bic = getattr(res, 'bic', float('inf'))
            if not math.isfinite(bic):
                return False

        # Guardarail Ljung-Box (opcional)
        if check_lb and not self._ljung_box_ok(res, lags=lb_lags, alpha=alpha, check_sq=check_sq):
            return False

        return True

    @staticmethod
    def _forecast_with_pi(
        res: Any,
        steps: int = 1,
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
             Genera pronóstico y bandas (1 - alpha) para un ARIMAResults (statsmodels).

            Retorna:
              mean  : np.ndarray de medias pronosticadas (longitud = steps)
              lower : np.ndarray con límites inferiores
              upper : np.ndarray con límites superiores
        """
    
        # get_forecast suele soportar alpha; si no, pedimos sin alpha y lo aplicamos en config_init  
        try:
            fc = res.get_forecast(steps=steps, alpha=alpha)
        except TypeError:
            fc = res.get_forecast(steps=steps)

        # Media pronosticada en array
        if hasattr(fc, 'predicted_mean'):
            mean = np.asarray(fc.predicted_mean)
        else:
            # fallback muy raro, pero puede ocurrir
            mean = np.asarray(getattr(fc, 'mean', np.array([])))

        # Intervalos de confianza: usar summary_frame si existe, si no conf_int()
        lower, upper = None, None
        if hasattr(fc, "summary_frame"):
            sf = fc.summary_frame(alpha=alpha)
            # nombres robustos: statsmodels usa mean_ci_lower/upper o lower y / upper y
            cols = {c.lower(): c for c in sf.columns}
            if "mean_ci_lower" in cols and "mean_ci_upper" in cols:
                lower = sf[cols["mean_ci_lower"]].to_numpy()
                upper = sf[cols["mean_ci_upper"]].to_numpy()
            else:
                # tomar 1ª y 2ª columnas como fallback
                lower = sf.iloc[:, 0].to_numpy()
                upper = sf.iloc[:, 1].to_numpy()
        else:
            ci = fc.conf_int(alpha=alpha)
            if isinstance(ci, pd.DataFrame):
                lower = ci.iloc[:, 0].to_numpy()
                upper = ci.iloc[:, 1].to_numpy()
            else:
                ci = np.asarray(ci)
                lower = ci[:, 0]
                upper = ci[:, 1]
    
        return mean, lower, upper

    @staticmethod
    def _minmax(s: pd.Series, eps: float = 1e-12) -> pd.Series:
        """
        Escala a [0,1] con min–max robusto:
          x' = (x - min) / max((max - min), eps)
        - Mantiene el índice original.
        - Si max == min, usa 'eps' para evitar división por 0.
        - Si todo es NaN o la serie está vacía, devuelve NaN/serie vacía correspondiente.
        """
        s = pd.to_numeric(s, errors="coerce").astype(float)
        if s.size == 0:
            return s
        vals = s.values
        mask = np.isfinite(vals)
        if not mask.any():  # todo NaN/inf
            return pd.Series(np.nan, index=s.index, name=s.name)

        smin = float(np.nanmin(vals))
        smax = float(np.nanmax(vals))
        denom = max(smax - smin, eps)
        return (s - smin) / denom
        
    # ------------------------ API pública ------------------------------

    def fit_grid(
        self,
        p_vals: Iterable[int] = range(0, 4),
        d_vals: Iterable[int] = range(0, 3),
        q_vals: Iterable[int] = range(0, 4),
        *,
        store_only_valid: bool = False,
        require_converged: bool = False,
        check_lb: bool = False,
        lb_lags: Iterable[int] = (10, 20, 40),
        alpha: float = 0.05,
        check_sq: bool = False,
    ) -> "AutoArimaExplorer":
        """
        Ajusta y almacena una rejilla de modelos ARIMA(p,d,q).
    
        Recorre todas las combinaciones de (p, d, q) indicadas y:
          1) Selecciona la serie interna apropiada a `d`
             - d = 0 → usa self.y_ret (log-retornos)
             - d ≥ 1 → usa self.y_lvls (niveles estandarizados)
          2) Comprueba un criterio mínimo de datos vía `_has_enough_data(...)`
          3) Ajusta el candidato con reintentos y elige el *mejor intento por BIC*
             mediante `_fit_candidate_insample((p,d,q))`
          4) Guarda el resultado (ARIMAResults) en `self.results[(p,d,q)]`
             y, si falló, registra el error en `self.errors[(p,d,q)]`.
    
        Parámetros
        ----------
        p_vals, d_vals, q_vals : Iterable[int]
            Rangos de órdenes AR, integración y MA a explorar.
            Por defecto, p=0..3, d=0..2, q=0..3.
    
        store_only_valid : bool, opcional (False por defecto)
            Si True, solo almacena en `self.results` los modelos que pasan `_result_ok(...)`
            (p.ej., convergencia, BIC finito y Ljung–Box si se solicita).
            Si False, almacena todos los modelos que consigan ajustarse (aunque luego
            puedas filtrarlos en `select_best_by_bic` o `stepwise_by_q`).
    
        require_converged : bool, opcional (False por defecto)
            Si `store_only_valid=True`, exige que el ajuste reporte convergencia.
    
        check_lb : bool, opcional (False por defecto)
            Si `store_only_valid=True`, activa el guardarraíl Ljung–Box en residuos
            (y opcionalmente residuos^2 con `check_sq`).
    
        lb_lags : Iterable[int], opcional
            Lags a usar en Ljung–Box si `check_lb=True`. Por defecto (10, 20, 40).
    
        alpha : float, opcional
            Nivel de significancia para Ljung–Box (p ≥ alpha ⇒ OK). Por defecto 0.05.
    
        check_sq : bool, opcional (False por defecto)
            Si True y `check_lb=True`, también verifica Ljung–Box sobre residuos^2
            (proxy de heterocedasticidad de corto plazo).
    
        Efectos secundarios
        -------------------
        - `self.results[(p,d,q)] = ModelSpec(...)` con el objeto de resultados (o `res=None` si falló).
        - `self.errors[(p,d,q)] = str` con el último error capturado cuando no se pudo ajustar.
    
        Devuelve
        --------
        self : AutoArimaExplorer
            Para encadenar llamadas (fluent API).
    
        Notas
        -----
        - El criterio de “suficiencia de datos” usa `_has_enough_data(n, p, d, q)`,
          que combina un mínimo absoluto y un múltiplo de parámetros (heurística Box–Jenkins).
        - El ajuste por cada (p,d,q) usa `_fit_candidate_insample`, que a su vez
          ejecuta varios *trials* (combinaciones de `trend`/restricciones/estimador)
          y elige el **mejor por BIC**.
        """
        for d in d_vals:
            y_use = self._series_for_d(d)
            n = len(y_use)
    
            for p in p_vals:
                for q in q_vals:
                    key = (p, d, q)
    
                    # 1) Protección por tamaño muestral (evita intentos inútiles)
                    if not self._has_enough_data(n, p, d, q):
                        self.errors[key] = f"insufficient data for (p={p}, d={d}, q={q}): n={n}"
                        continue
    
                    # 2) Ajuste con política de reintentos y selección por BIC
                    res, err = self._fit_candidate_insample(order=key)
    
                    # 3) Almacenamiento y (opcional) filtrado de calidad
                    if res is not None:
                        if store_only_valid:
                            # Aplica filtros de calidad homogéneos (_result_ok centraliza reglas)
                            if not self._result_ok(
                                res,
                                require_converged=require_converged,
                                require_finite_bic=True,
                                check_lb=check_lb,
                                lb_lags=lb_lags,
                                alpha=alpha,
                                check_sq=check_sq,
                            ):
                                # No pasa filtros: lo registramos como error y no guardamos el res
                                self.errors[key] = "failed _result_ok filters"
                                # También guardamos una ficha vacía para mantener trazabilidad de la rejilla
                                self.results[key] = ModelSpec(p=p, d=d, q=q, res=None)
                                continue
    
                        # OK para guardar
                        self.results[key] = ModelSpec(p=p, d=d, q=q, res=res)
                    else:
                        # Falló el ajuste: registra error y ficha vacía
                        self.results[key] = ModelSpec(p=p, d=d, q=q, res=None)
                        self.errors[key] = err or "unknown error"
    
                    # (Opcional) logging por cada intento
                    self.logger.debug(
                        "fit_grid: (p=%s,d=%s,q=%s) -> %s",
                        p, d, q, "OK" if (res is not None) else f"FAIL: {self.errors.get(key)}"
                    )
    
        return self


    def select_best_by_bic(
        self,
        *,
        require_lb_guardrail: bool = True,
        lb_lags: Iterable[int] = (10, 20, 40),
        alpha: float = 0.05,
        check_sq: bool = False,
        require_converged: bool = False,
        prefer_simpler_on_ties: bool = True,
    ) -> Optional[Tuple[Tuple[int, int, int], ModelSpec]]:
        """
        Selecciona el mejor modelo por **BIC** entre los ya ajustados en `self.results`.
    
        Flujo
        -----
        1) Recorre `self.results[(p,d,q)]` y filtra:
           - que exista objeto de resultados (`res is not None`)
           - (opcional) `converged == True` si `require_converged=True`
           - BIC finito (descarta NaN/±inf)
           - (opcional) guardarraíl de Ljung–Box sobre residuos (y residuos^2 si `check_sq=True`)
        2) Entre los válidos, elige el **mínimo por BIC**.
        3) En caso de empates numéricos, puede preferir el más **parsonoioso** (`p+q` menor)
           si `prefer_simpler_on_ties=True`.
    
        Parámetros
        ----------
        require_lb_guardrail : bool, opcional (True)
            Si True, exige pasar Ljung–Box en los lags indicados (`lb_lags`) con nivel `alpha`.
            Si además `check_sq=True`, también exige Ljung–Box OK en residuos^2.
        lb_lags : Iterable[int], opcional
            Lags para Ljung–Box. Por defecto (10, 20, 40).
        alpha : float, opcional
            Nivel de significancia en Ljung–Box (p ≥ alpha ⇒ OK). Por defecto 0.05.
        check_sq : bool, opcional (False)
            Si True, además de residuos, comprueba Ljung–Box sobre residuos^2 (detectar ARCH corto plazo).
        require_converged : bool, opcional (False)
            Si True, descarta modelos cuya optimización no reporta convergencia.
        prefer_simpler_on_ties : bool, opcional (True)
            Si True, en empates por BIC usa un desempate de parsimonia: primero menor (p+q),
            después menor p y después menor q.
    
        Devuelve
        --------
        (best_key, best_spec) o None
            - `best_key` = (p, d, q)
            - `best_spec` = ModelSpec con `res` (ARIMAResultsWrapper)
            - `None` si ningún modelo pasó los filtros.
    
        Notas
        -----
        - Debes haber llamado antes a `fit_grid(...)` (o haber poblado `self.results`) para que
          exista algo que seleccionar.
        - Esta función **no re-ajusta** modelos; solo **selecciona** entre los ya guardados.
        """
        valid: List[Tuple[Tuple[int, int, int], ModelSpec]] = []
    
        for key, spec in self.results.items():
            res = spec.res
            if res is None:
                continue  # no hay ajuste
    
            # 1) Convergencia (si se exige)
            if require_converged and getattr(res, "converged", True) is False:
                continue
    
            # 2) BIC finito (evita soluciones patológicas o mal condicionadas)
            bic = getattr(res, "bic", float("inf"))  # si no hay BIC -> +inf
            if not math.isfinite(bic):
                continue
    
            # 3) Guardarraíl Ljung–Box (media y opcionalmente varianza)
            if require_lb_guardrail:
                if not self._ljung_box_ok(res, lags=lb_lags, alpha=alpha, check_sq=check_sq):
                    continue
    
            valid.append((key, spec))
    
        if not valid:
            return None  # ningún candidato pasó los filtros
    
        if prefer_simpler_on_ties:
            # Orden: (BIC, p+q, p, q) → favorece el más parsimonioso en empate de BIC
            def _key(kv):
                (p, d, q), spec = kv
                return (spec.res.bic, p + q, p, q)
            best_key, best_spec = min(valid, key=_key)
        else:
            # Solo por BIC
            best_key, best_spec = min(valid, key=lambda kv: kv[1].res.bic)
    
        return best_key, best_spec

    def select_by_q(
        self,
        *,
        d_vals: Optional[Iterable[int]] = None,
        q_vals: Optional[Iterable[int]] = None,
        require_lb_guardrail: bool = True,
        lb_lags: Iterable[int] = (10, 20, 40),
        alpha: float = 0.05,
        check_sq: bool = False,
        require_converged: bool = False,
        prefer_simpler_on_ties: bool = True,
    ) -> Dict[int, Dict[int, Tuple[int, ModelSpec]]]:
        """
        Para cada (d, q) elige el mejor ARIMA(p, d, q) por **BIC** entre los modelos ya ajustados.
    
        Flujo
        -----
        1) Determina d_vals y q_vals a partir de `self.results` si no se pasan explícitos.
        2) Para cada (d, q), construye la lista de candidatos {p} con resultados válidos:
           - `res is not None`
           - (opcional) `converged == True` si `require_converged=True`
           - BIC finito
           - (opcional) guardarraíl Ljung–Box en residuos (y en residuos² si `check_sq=True`)
        3) Selecciona el **mínimo por BIC**. En empates, puede preferir el más parsimonioso
           (`p+q` menor; luego menor `p`, luego `q`) si `prefer_simpler_on_ties=True`.
    
        Parámetros
        ----------
        d_vals, q_vals : Iterable[int] o None
            Conjuntos de d y q a evaluar. Si None, se infieren de las claves de `self.results`.
        require_lb_guardrail : bool
            Exigir Ljung–Box OK en `lb_lags` al nivel `alpha`. Si `check_sq=True`, también
            se exige en residuos² (detecta ARCH de corto plazo).
        lb_lags : Iterable[int]
            Lags usados en Ljung–Box. Por defecto (10, 20, 40).
        alpha : float
            Nivel de significancia de Ljung–Box. p ≥ alpha ⇒ OK.
        check_sq : bool
            Si True, añade Ljung–Box sobre residuos².
        require_converged : bool
            Si True, descarta modelos con `converged=False`.
        prefer_simpler_on_ties : bool
            Si True, en empates por BIC prioriza menor (p+q), luego menor p y luego menor q.
    
        Devuelve
        --------
        winners : dict
            Diccionario anidado `{ d: { q: (p_elegido, ModelSpec) } }`.
            Entradas (d,q) sin candidatos válidos no aparecen.
    
        Notas
        -----
        - No re-ajusta modelos; solo selecciona entre los ya presentes en `self.results`.
        - Llama a `fit_grid(...)` antes para poblar `self.results`.
        """
        # 0) Inferir d_vals / q_vals si no se proporcionan
        if d_vals is None:
            d_vals = sorted({d for (_, d, _) in self.results.keys()})
        else:
            d_vals = list(d_vals)
    
        if q_vals is None:
            q_vals = sorted({q for (_, _, q) in self.results.keys()})
        else:
            q_vals = list(q_vals)
    
        winners: Dict[int, Dict[int, Tuple[int, ModelSpec]]] = {}
    
        for d in d_vals:
            # Valores de p disponibles para este d,q se obtendrán por q
            winners[d] = {}
    
            for q in q_vals:
                # Construir cadena de candidatos (p, spec) para este (d, q)
                candidates: List[Tuple[int, ModelSpec]] = []
                # todos los p presentes para este (d,q)
                p_vals = sorted({p for (p, d2, q2) in self.results.keys() if d2 == d and q2 == q})
    
                for p in p_vals:
                    key = (p, d, q)
                    spec = self.results.get(key)
                    if spec is None or spec.res is None:
                        continue
    
                    res = spec.res
    
                    # 1) Convergencia (si se exige)
                    if require_converged and getattr(res, "converged", True) is False:
                        continue
    
                    # 2) BIC finito
                    bic = getattr(res, "bic", float("inf"))
                    if not math.isfinite(bic):
                        continue
    
                    # 3) Guardarraíl Ljung–Box (media y opcionalmente varianza)
                    if require_lb_guardrail:
                        if not self._ljung_box_ok(res, lags=lb_lags, alpha=alpha, check_sq=check_sq):
                            continue
    
                    candidates.append((p, spec))
    
                if not candidates:
                    # Nada válido para este (d, q)
                    continue
    
                # Selección: mínimo por BIC con posible desempate por parsimonia
                if prefer_simpler_on_ties:
                    def _tie_key(item: Tuple[int, ModelSpec]):
                        p_i, spec_i = item
                        # p_i == spec_i.p y q == spec_i.q; usamos ambos por claridad
                        return (spec_i.res.bic, spec_i.p + spec_i.q, spec_i.p, spec_i.q)
                    p_best, spec_best = min(candidates, key=_tie_key)
                else:
                    p_best, spec_best = min(candidates, key=lambda it: it[1].res.bic)
    
                winners[d][q] = (p_best, spec_best)
    
                # Log resumen por (d,q)
                try:
                    self.logger.info(
                        "[select_by_q] d=%s q=%s → elegido ARIMA(%s,%s,%s) | BIC=%.3f",
                        d, q, p_best, d, q, spec_best.res.bic
                    )
                except Exception:
                    pass
    
        return winners

    def select_by_oos(
        self,
        *,
        candidate_keys: Optional[Iterable[Tuple[int, int, int]]] = None,
        horizon: int = 1,
        initial: Optional[int] = None,
        step: int = 1,
        max_folds: Optional[int] = 20,
        metric: str = "rmse",     # 'rmse' | 'mae' | 'mape' | 'winkler'
        scale_target: bool = False,
        require_converged_fold: bool = True,
        lb_guardrail_on_folds: bool = False,
        lb_lags: Iterable[int] = (10, 20, 40),
        alpha: float = 0.05,      # para Winkler / PI
        check_sq: bool = False,   # LB sobre residuos^2 si se activa guardarraíl
    ) -> Optional[Tuple[Tuple[int, int, int], ModelSpec, Dict[Tuple[int, int, int], float]]]:
        """
        Selección **out-of-sample** (OOS) por validación rolling-origin.
    
        Para cada modelo candidato (p,d,q):
          - Toma la serie interna adecuada a d (retornos si d=0; niveles estandarizados si d≥1).
          - Realiza una validación OOS "origen rodante":
              en cada fold t:
                * Ajusta el modelo sobre y[:t] (usando _fit_candidate_on_subset).
                * Pronostica `horizon` pasos.
                * Calcula el error OOS con la métrica elegida (RMSE/MAE/MAPE) o
                  el **Winkler score** si `metric='winkler'` (requiere PI).
          - Agrega los errores de todos los folds (media) → score OOS del modelo.
        Devuelve el modelo con **menor score**.
    
        Parámetros
        ----------
        candidate_keys : Iterable[(p,d,q)] o None
            Conjunto de candidatos. Si None, usa todas las claves presentes en self.results.
        horizon : int
            Horizonte de pronóstico por fold (≥1).
        initial : int o None
            Tamaño mínimo de ventana de entrenamiento. Si None:
              max(50, 10*(p+q+d+1)).
        step : int
            Salto entre folds (1 = evaluar cada punto).
        max_folds : int o None
            Límite superior de folds evaluados (desde el final hacia atrás). Útil para acelerar.
        metric : str
            'rmse' | 'mae' | 'mape' | 'winkler'.
            - rmse/mae/mape usan la media pronosticada.
            - winkler usa PI (1-alpha) y la métrica `_winkler_score`.
        scale_target : bool
            Si True, escala la serie y a [0,1] con `_minmax` antes de evaluar (comparable entre d).
        require_converged_fold : bool
            Si True, descarta un fold cuyo ajuste no converge (penaliza con NaN/inf).
        lb_guardrail_on_folds : bool
            Si True, exige que cada fold pase Ljung–Box (media y opcionalmente varianza con `check_sq`).
        lb_lags, alpha, check_sq :
            Parámetros del guardarraíl de Ljung–Box (igual que en otros métodos).
    
        Devuelve
        --------
        (best_key, best_spec, scores) o None
            - best_key  : (p, d, q) del ganador OOS
            - best_spec : ModelSpec re-ajustado in-sample completo (para operar luego)
            - scores    : dict {(p,d,q): score_OOS}
            - None si no hubo ningún candidato evaluable.
    
        Notas
        -----
        - Este método **re-ajusta** en cada fold (subserie) por cada candidato → puede ser costoso.
        - Para acelerar, usa `max_folds`, `step` > 1, y/o acota `candidate_keys`.
        """
        
        # --- preparar candidatos ---
        if candidate_keys is None:
            candidate_keys = list(self.results.keys())
        else:
            candidate_keys = list(candidate_keys)
    
        # Helpers de métricas (vectorizados)
        def _rmse(y, yhat) -> float:
            y = np.asarray(y, float); yhat = np.asarray(yhat, float)
            return float(np.sqrt(np.mean((y - yhat) ** 2)))
    
        def _mae(y, yhat) -> float:
            y = np.asarray(y, float); yhat = np.asarray(yhat, float)
            return float(np.mean(np.abs(y - yhat)))
    
        def _mape(y, yhat, eps: float = 1e-12) -> float:
            y = np.asarray(y, float); yhat = np.asarray(yhat, float)
            denom = np.maximum(np.abs(y), eps)
            return float(np.mean(np.abs((y - yhat) / denom)))
    
        use_winkler = (metric.lower() == "winkler")
        metric = metric.lower()
    
        scores: Dict[Tuple[int, int, int], float] = {}
    
        for (p, d, q) in candidate_keys:
            # Serie adecuada a d
            y_full = self._series_for_d(d).copy()
            if scale_target:
                y_full = self._minmax(y_full)
    
            n = len(y_full)
            if n <= horizon + 5:
                scores[(p, d, q)] = float("inf")
                continue
    
            # Tamaño mínimo de train (por defecto: heurística)
            min_train = initial if (initial is not None) else max(50, 10 * (p + q + d + 1))
            min_train = min_train + 0  # ensure int
            if min_train >= n - horizon:
                # imposible hacer al menos un fold
                scores[(p, d, q)] = float("inf")
                continue
    
            # Folds: t es el tamaño de train (excluye los horizon valores de test)
            t_start = min_train
            t_end = n - horizon
            t_grid = list(range(t_start, t_end + 1, step))
            # para acelerar: coge los últimos max_folds
            if (max_folds is not None) and (len(t_grid) > max_folds):
                t_grid = t_grid[-max_folds:]
    
            fold_scores: List[float] = []
    
            for t in t_grid:
                y_train = y_full.iloc[:t]
                y_true = y_full.iloc[t : t + horizon].to_numpy()
    
                # Ajuste en subserie
                res, err = self._fit_candidate_on_subset(y_train, (p, d, q))
                if res is None:
                    fold_scores.append(float("inf"))
                    continue
    
                # Filtrado por fold (convergencia / LB opcional)
                if require_converged_fold and getattr(res, "converged", True) is False:
                    fold_scores.append(float("inf"))
                    continue
                if lb_guardrail_on_folds:
                    if not self._ljung_box_ok(res, lags=lb_lags, alpha=alpha, check_sq=check_sq):
                        fold_scores.append(float("inf"))
                        continue
    
                # Pronóstico y score del fold
                try:
                    mean, lower, upper = self._forecast_with_pi(res, steps=horizon, alpha=alpha)
                except Exception:
                    fold_scores.append(float("inf"))
                    continue
    
                if use_winkler:
                    # Winkler por paso y promedio en el horizonte
                    sc = 0.0
                    for i in range(len(y_true)):
                        sc += self._winkler_score(float(y_true[i]), float(lower[i]), float(upper[i]), alpha=alpha)
                    fold_scores.append(sc / len(y_true))
                else:
                    if metric == "rmse":
                        fold_scores.append(_rmse(y_true, mean))
                    elif metric == "mae":
                        fold_scores.append(_mae(y_true, mean))
                    elif metric == "mape":
                        fold_scores.append(_mape(y_true, mean))
                    else:
                        # métrica desconocida → penaliza
                        fold_scores.append(float("inf"))
    
            # Score OOS del modelo: media de folds (ignorando NaN)
            sc = float(np.nanmean(fold_scores)) if len(fold_scores) else float("inf")
            scores[(p, d, q)] = sc
    
            # logging
            try:
                self.logger.info(
                    "[OOS] ARIMA(%s,%s,%s) | metric=%s | horizon=%s | folds=%s | score=%.6f",
                    p, d, q, metric, horizon, len(fold_scores), sc
                )
            except Exception:
                pass
    
        if not scores:
            return None
    
        # Elegir mejor por score OOS
        best_key = min(scores.items(), key=lambda kv: kv[1])[0]
        if not math.isfinite(scores[best_key]):
            # nadie fue evaluable realmente
            return None
    
        # Re-ajustar in-sample completo el ganador para devolver un ModelSpec utilizable
        res_full, err_full = self._fit_candidate_insample(order=best_key)
        if res_full is None:
            return None
    
        best_spec = ModelSpec(p=best_key[0], d=best_key[1], q=best_key[2], res=res_full)
        return best_key, best_spec, scores