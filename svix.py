import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import BSpline
from scipy.optimize import minimize_scalar, brentq
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.linalg import cholesky, solve
import warnings
from datetime import datetime, timedelta
try:
    import yfinance as yf
except ImportError:
    yf = None
    print("Warning: yfinance not installed. Please install via 'pip install yfinance'")
try:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
except ImportError:
    plt = None
    print('Warning: matplotlib not installed.')
warnings.filterwarnings('ignore')

class YieldCurve:

    def __init__(self):
        self.tickers = {0.25: '^IRX', 5.0: '^FVX', 10.0: '^TNX', 30.0: '^TYX'}
        self.rates = {}
        self.curve_func = None
        self.refresh()

    def refresh(self):
        print('Fetching Treasury Yields from yfinance...')
        try:
            data = yf.download(list(self.tickers.values()), period='5d', progress=False)['Close']
            times = []
            yields = []
            latest = data.iloc[-1]
            if latest.isnull().any():
                latest = data.iloc[-2]
            for T, ticker in self.tickers.items():
                val = latest[ticker]
                r = val / 100.0
                self.rates[T] = r
                times.append(T)
                yields.append(r)
                print(f'  {ticker} ({T} yr): {r:.2%}')
            from scipy.interpolate import interp1d
            self.curve_func = interp1d(times, yields, kind='linear', fill_value='extrapolate')
        except Exception as e:
            print(f'Error fetching yields: {e}. Defaulting to 4% flat.')
            self.curve_func = lambda x: 0.04

    def get_rate(self, time_to_maturity):
        t = max(time_to_maturity, 7 / 365.0)
        return float(self.curve_func(t))

class PenalizedBSpline:

    def __init__(self, x, y, w=None, n_knots=30, degree=3, lam=1.0):
        self.degree = degree
        idx = np.argsort(x)
        self.x_data = x[idx]
        self.y_data = y[idx]
        self.w_data = w[idx] if w is not None else np.ones_like(x)
        x_min, x_max = (self.x_data[0], self.x_data[-1])
        dx = (x_max - x_min) / (n_knots - 1)
        self.knots = np.linspace(x_min - degree * dx, x_max + degree * dx, n_knots + 2 * degree)
        n_coeffs = len(self.knots) - (degree + 1)
        self.n_coeffs = n_coeffs
        self.B = np.zeros((len(self.x_data), n_coeffs))
        for i in range(n_coeffs):
            c_temp = np.zeros(n_coeffs)
            c_temp[i] = 1.0
            spl = BSpline(self.knots, c_temp, degree)
            self.B[:, i] = spl(self.x_data)
        D = np.zeros((n_coeffs - 2, n_coeffs))
        for i in range(n_coeffs - 2):
            D[i, i] = 1
            D[i, i + 1] = -2
            D[i, i + 2] = 1
        W_diag = np.diag(self.w_data)
        BtW = self.B.T @ W_diag
        BtWB = BtW @ self.B
        DTD = D.T @ D
        LHS = BtWB + lam * DTD
        RHS = BtW @ self.y_data
        try:
            self.c = solve(LHS, RHS, assume_a='pos')
        except np.linalg.LinAlgError:
            self.c = np.linalg.lstsq(LHS, RHS, rcond=None)[0]
        self.spline_model = BSpline(self.knots, self.c, degree)

    def predict(self, x_new):
        return self.spline_model(x_new)

    def predict_derivative(self, x_new, order=1):
        return self.spline_model.derivative(order)(x_new)

class RobustSVIXCalculator:

    def __init__(self, risk_free_rate, time_to_maturity):
        self.r = risk_free_rate
        self.T = max(time_to_maturity, 0.001)
        self.df = np.exp(-self.r * self.T)
        self.Rf_gross = np.exp(self.r * self.T)

    def _black_76_price(self, F, K, sigma, option_type='call'):
        sigma = np.maximum(sigma, 0.0001)
        d1 = (np.log(F / K) + 0.5 * sigma ** 2 * self.T) / (sigma * np.sqrt(self.T))
        d2 = d1 - sigma * np.sqrt(self.T)
        if option_type == 'call':
            return self.df * (F * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            return self.df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

    def _get_analytic_pdf(self, F, K_grid, sigma, sigma_k, sigma_kk):
        T = self.T
        sqT = np.sqrt(T)
        d1 = (np.log(F / K_grid) + 0.5 * sigma ** 2 * T) / (sigma * sqT)
        d2 = d1 - sigma * sqT
        n_d1 = norm.pdf(d1)
        n_d2 = norm.pdf(d2)
        C_KK = self.df * n_d2 / (K_grid * sigma * sqT)
        C_sigma = F * self.df * n_d1 * sqT
        C_vomma = C_sigma * d1 * d2 / sigma
        C_vanna = self.df * n_d2 * (d1 / sigma)
        pdf = C_KK + 2 * C_vanna * sigma_k + C_vomma * sigma_k ** 2 + C_sigma * sigma_kk
        return pdf * self.Rf_gross

    def _implied_volatility(self, price, F, K, option_type='call'):
        intrinsic = max(0, F - K) * self.df if option_type == 'call' else max(0, K - F) * self.df
        if price <= intrinsic + 0.001:
            return np.nan

        def obj(s):
            return self._black_76_price(F, K, s, option_type) - price
        try:
            return brentq(obj, 0.001, 6.0)
        except:
            return np.nan

    def fit_vol_surface(self, chain, spot_ref, lam_smoothing=0.5):
        valid = chain[(chain['call_price'] > 0.05) & (chain['put_price'] > 0.05)].copy()
        F_est = spot_ref * self.Rf_gross
        if not valid.empty:
            valid['diff'] = abs(valid['strike'] - spot_ref)
            atm_subset = valid.sort_values('diff').head(5)
            f_estimates = atm_subset['strike'] + (atm_subset['call_price'] - atm_subset['put_price']) / self.df
            F = f_estimates.median()
        else:
            F = F_est
        strikes, ivs, weights = ([], [], [])
        for i, row in chain.iterrows():
            K = row['strike']
            if K > F * 5.0 or K < F * 0.15:
                continue
            if K < F:
                price, typ = (row['put_price'], 'put')
            else:
                price, typ = (row['call_price'], 'call')
            if price <= 0:
                continue
            vol = self._implied_volatility(price, F, K, typ)
            if not np.isnan(vol) and 0.01 < vol < 5.0:
                strikes.append(K)
                ivs.append(vol)
                w = np.exp(-1.0 * abs(np.log(K / F)))
                weights.append(w)
        strikes = np.array(strikes)
        ivs = np.array(ivs)
        weights = np.array(weights)
        if len(strikes) < 5:
            avg_iv = np.mean(ivs) if len(ivs) > 0 else 0.4
            dummy_spline = lambda x: avg_iv
            return (dummy_spline, None, F, F * 0.8, F * 1.2, (strikes, ivs))
        idx = np.argsort(strikes)
        strikes = strikes[idx]
        ivs = ivs[idx]
        weights = weights[idx]
        min_k_data, max_k_data = (strikes[0], strikes[-1])
        log_ks = np.log(strikes / F)
        n_knots = max(10, min(50, int(len(strikes) / 1.5)))
        p_spline = PenalizedBSpline(log_ks, ivs, weights, n_knots=n_knots, lam=lam_smoothing)

        def vol_func(k_strike, return_derivs=False):
            k_strike = np.atleast_1d(k_strike)
            x_query = np.log(k_strike / F)
            x_min = np.log(min_k_data / F)
            x_max = np.log(max_k_data / F)
            vols = np.zeros_like(x_query)
            d_vols = np.zeros_like(x_query)
            d2_vols = np.zeros_like(x_query)
            mask_in = (x_query >= x_min) & (x_query <= x_max)
            if np.any(mask_in):
                vols[mask_in] = p_spline.predict(x_query[mask_in])
                if return_derivs:
                    d_vols[mask_in] = p_spline.predict_derivative(x_query[mask_in], 1)
                    d2_vols[mask_in] = p_spline.predict_derivative(x_query[mask_in], 2)
            mask_left = x_query < x_min
            if np.any(mask_left):
                val_min = p_spline.predict(x_min)
                slope_min = p_spline.predict_derivative(x_min, order=1)
                vols[mask_left] = val_min + slope_min * (x_query[mask_left] - x_min)
                if return_derivs:
                    d_vols[mask_left] = slope_min
                    d2_vols[mask_left] = 0.0
            mask_right = x_query > x_max
            if np.any(mask_right):
                val_max = p_spline.predict(x_max)
                slope_max = p_spline.predict_derivative(x_max, order=1)
                vols[mask_right] = val_max + slope_max * (x_query[mask_right] - x_max)
                if return_derivs:
                    d_vols[mask_right] = slope_max
                    d2_vols[mask_right] = 0.0
            vols = np.maximum(vols, 0.01)
            if return_derivs:
                return (vols, d_vols, d2_vols)
            return vols
        return (vol_func, p_spline, F, min_k_data, max_k_data, (strikes, ivs))

    def compute_svix(self, chain, spot_display, manual_smoothing=None):
        lam = manual_smoothing if manual_smoothing is not None else 0.5
        vol_func, p_spline_obj, F, min_k, max_k, raw_data = self.fit_vol_surface(chain, spot_display, lam)
        grid_low = np.linspace(F * 0.01, F * 0.8, 500)
        grid_mid = np.linspace(F * 0.8, F * 1.2, 1000)
        grid_high = np.linspace(F * 1.2, F * 6.0, 1333)
        K_grid = np.unique(np.concatenate([grid_low, grid_mid, grid_high]))
        K_grid = K_grid[K_grid > 0.001]
        sigmas, d_sigmas_x, d2_sigmas_x = vol_func(K_grid, return_derivs=True)
        if p_spline_obj is not None:
            sigma_k = d_sigmas_x / K_grid
            sigma_kk = (d2_sigmas_x - d_sigmas_x) / K_grid ** 2
            pdf_values = self._get_analytic_pdf(F, K_grid, sigmas, sigma_k, sigma_kk)
        else:
            call_prices = self._black_76_price(F, K_grid, sigmas, 'call')
            pdf_values = np.zeros_like(call_prices)
        pdf_values = np.maximum(pdf_values, 0.0)
        area = trapezoid(pdf_values, K_grid)
        if area > 0:
            pdf_values /= area
        otm_prices = np.zeros_like(K_grid)
        is_put = K_grid < F
        is_call = ~is_put
        if np.any(is_put):
            otm_prices[is_put] = self._black_76_price(F, K_grid[is_put], sigmas[is_put], 'put')
        if np.any(is_call):
            otm_prices[is_call] = self._black_76_price(F, K_grid[is_call], sigmas[is_call], 'call')
        integral_val = trapezoid(otm_prices, K_grid)
        svix_annual_var = 2 * self.Rf_gross * integral_val / (self.T * F ** 2)
        return (svix_annual_var, raw_data, (K_grid, sigmas), (K_grid, pdf_values), F)

class ExpectedReturnModel:

    def __init__(self, risk_free_rate, time_to_maturity):
        self.calc = RobustSVIXCalculator(risk_free_rate, time_to_maturity)

    def calculate(self, df_stock, spot_display, df_market=None, market_spot=None, is_index=False, manual_smoothing=None):
        svix_stock_ann, raw_stock, curve_stock, pdf_stock, F_stock = self.calc.compute_svix(df_stock, spot_display, manual_smoothing)
        if is_index:
            svix_market_ann = svix_stock_ann
        elif df_market is not None:
            svix_market_ann, _, _, _, _ = self.calc.compute_svix(df_market, market_spot)
        else:
            svix_market_ann = 0.04
        if is_index:
            period_variance = svix_stock_ann * self.calc.T
        else:
            period_variance = (0.5 * svix_market_ann + 0.5 * svix_stock_ann) * self.calc.T
        E_ST = F_stock * (1 + period_variance)
        exp_ann_ret = (E_ST / spot_display) ** (1 / self.calc.T) - 1
        return {'T': self.calc.T, 'expected_return_ann': exp_ann_ret, 'svix_annual_var': svix_stock_ann, 'debug_pdf': pdf_stock, 'debug_curve': curve_stock, 'debug_raw': raw_stock, 'spot': spot_display, 'fwd': F_stock, 'E_ST': E_ST, 'rf_gross': self.calc.Rf_gross}

def get_data(ticker):
    if yf is None:
        return (None, 100, None)
    tk = yf.Ticker(ticker)
    try:
        spot = tk.fast_info['last_price']
    except:
        spot = tk.history(period='1d')['Close'].iloc[-1]
    hist = tk.history(period='6mo')
    return (tk, spot, hist)

def get_chain(tk, date):
    opt = tk.option_chain(date)
    calls = opt.calls[['strike', 'bid', 'ask', 'lastPrice']].copy()
    puts = opt.puts[['strike', 'bid', 'ask', 'lastPrice']].copy()

    def mid(r):
        return (r['bid'] + r['ask']) / 2 if r['bid'] > 0 else r['lastPrice']
    calls['call_price'] = calls.apply(mid, axis=1)
    puts['put_price'] = puts.apply(mid, axis=1)
    df = pd.merge(calls[['strike', 'call_price']], puts[['strike', 'put_price']], on='strike', how='outer')
    return df.sort_values('strike')

class AnalysisDashboard:

    def __init__(self, results, history, ticker, rf_rate):
        self.results, self.history, self.ticker = (results, history, ticker)
        self.rf_rate = rf_rate
        self.current_idx = 0
        self.fig = plt.figure(figsize=(16, 10))
        gs = self.fig.add_gridspec(2, 2)
        self.ax_fan = self.fig.add_subplot(gs[0, :])
        self.ax_iv = self.fig.add_subplot(gs[1, 0])
        self.ax_pdf = self.fig.add_subplot(gs[1, 1])
        plt.subplots_adjust(bottom=0.2, top=0.95, hspace=0.3)
        self.s_date = Slider(plt.axes([0.2, 0.1, 0.6, 0.03]), 'Exp', 0, len(results) - 1, valinit=0, valstep=1)
        self.s_smooth = Slider(plt.axes([0.2, 0.05, 0.6, 0.03]), 'Smooth (Lambda)', 0.01, 500.0, valinit=1.0)
        self.s_date.on_changed(self.on_date_change)
        self.s_smooth.on_changed(self.on_smooth_change)
        self.plot_fan()
        self.refresh_plots()

    def run_fit(self, idx, lam):
        d = self.results[idx]
        is_ind = self.ticker in ['^SPX', 'SPY', '^NDX', 'QQQ']
        df_mkt = d.get('market_chain_df')
        spot_mkt = d.get('market_spot')
        new = ExpectedReturnModel(self.rf_rate, d['T']).calculate(d['chain_df'], d['spot'], is_index=is_ind, df_market=df_mkt, market_spot=spot_mkt, manual_smoothing=lam)
        new.update({'date': d['date'], 'chain_df': d['chain_df'], 'market_chain_df': df_mkt, 'market_spot': spot_mkt})
        self.results[idx] = new

    def on_date_change(self, val):
        self.current_idx = int(self.s_date.val)
        current_lambda = self.s_smooth.val
        self.run_fit(self.current_idx, current_lambda)
        self.refresh_plots()

    def on_smooth_change(self, val):
        for i in range(len(self.results)):
            self.run_fit(i, val)
        self.refresh_plots()

    def solve_physical(self, grid, q_pdf, target_E_ST, F):
        x = grid / F
        target_mean_x = target_E_ST / F

        def obj(gamma):
            try:
                w = q_pdf * np.power(x, gamma)
                norm_w = trapezoid(w, x)
                if norm_w == 0:
                    return 1000000000.0
                mean_x = trapezoid(x * w, x) / norm_w
                return mean_x - target_mean_x
            except:
                return 1000000000.0
        try:
            gamma = brentq(obj, -50, 50)
        except:
            gamma = 1.0
            print('Failed to find gamma, using default 1.0')
        w = q_pdf * x ** gamma
        p_pdf = w / trapezoid(w, grid)
        return (p_pdf, gamma)

    def compute_allocation(self, r_grid, p_pdf, rf_gross):
        rf_net = rf_gross - 1.0

        def get_expected_utility(f):
            wealth_grid = rf_gross + f * (r_grid - rf_net)
            if np.any(wealth_grid <= 1e-09):
                return -np.inf
            utility_grid = -np.reciprocal(wealth_grid)
            expected_util = trapezoid(p_pdf * utility_grid, r_grid)
            return expected_util

        def objective(f):
            return -get_expected_utility(f)
        r_min = r_grid[0]
        if r_min - rf_net < 0:
            max_lev_theoretical = -rf_gross / (r_min - rf_net)
            upper_bound = min(4.0, max_lev_theoretical * 0.99)
        else:
            upper_bound = 4.0
        try:
            res = minimize_scalar(objective, bounds=(0.0, upper_bound), method='bounded')
            best_f = res.x
            max_util = -res.fun
        except Exception as e:
            print(f'Optimization failed: {e}')
            best_f = 0.0
            max_util = 0.0
        return (best_f, max_util)

    def plot_fan(self):
        ax = self.ax_fan
        ax.clear()
        if self.history is not None:
            ax.plot(self.history.index, self.history['Close'], 'k', label='History')
            last_date = self.history.index[-1].replace(tzinfo=None)
            last_price = self.history['Close'].iloc[-1]
        else:
            last_date = datetime.now()
            last_price = self.results[0]['spot']
        fan = {p: [last_price] for p in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]}
        dates = [last_date]
        for res in sorted(self.results, key=lambda x: x['T']):
            dates.append(datetime.strptime(res['date'], '%Y-%m-%d'))
            k, q_dens = res['debug_pdf']
            F = res['fwd']
            E_ST = res['E_ST']
            p_pdf, _ = self.solve_physical(k, q_dens, E_ST, F)
            cdf = cumulative_trapezoid(p_pdf, k, initial=0)
            cdf /= cdf[-1]
            percs = np.interp([p / 100 for p in fan], cdf, k)
            for p, val in zip(fan, percs):
                fan[p].append(val)
        colors = ['purple'] * 6
        alphas = [0.1, 0.15, 0.2, 0.25, 0.3]
        for i, (low, high) in enumerate([(5, 95), (10, 90), (20, 80), (30, 70), (40, 60)]):
            ax.fill_between(dates, fan[low], fan[high], color=colors[i], alpha=alphas[i], label=f'{low}-{high}%' if i == 0 else '')
        ax.plot(dates, fan[50], 'purple', ls='--', label='Median')
        ax.set_title(f'Forecast Fan Chart: {self.ticker}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')

    def refresh_plots(self):
        d = self.results[self.current_idx]
        self.ax_iv.clear()
        F = d['fwd']
        self.ax_iv.plot(d['debug_raw'][0], d['debug_raw'][1], 'rx', alpha=0.5, label='Mkt')
        self.ax_iv.plot(d['debug_curve'][0], d['debug_curve'][1], 'b', lw=2, label='P-Spline')
        self.ax_iv.set_title(f'IV Structure (F={F:.2f})')
        self.ax_iv.legend()
        self.ax_iv.grid(True, alpha=0.3)
        self.ax_iv.set_xlabel('Strike')
        self.ax_pdf.clear()
        k, q_dens_price = d['debug_pdf']
        E_ST = d['E_ST']
        Spot = d['spot']
        p_dens_price, gamma = self.solve_physical(k, q_dens_price, E_ST, F)
        r_grid_spot = k / Spot - 1
        p_dens_ret = p_dens_price * Spot
        self.ax_pdf.plot(r_grid_spot, p_dens_ret, 'k', label=f'P (g={gamma:.1f})')
        self.ax_pdf.fill_between(r_grid_spot, 0, p_dens_ret, color='green', alpha=0.1)
        self.ax_pdf.set_title(f"Return Distribution {d['date']}")
        self.ax_iv.legend()
        self.ax_iv.grid(True, alpha=0.3)
        mean_ret = E_ST / Spot - 1
        self.ax_pdf.axvline(mean_ret, color='green', ls='--', label='Mean')
        self.ax_pdf.text(mean_ret, 0.95, f'{mean_ret:.1%}', transform=self.ax_pdf.get_xaxis_transform(), color='green', ha='left')
        rf_g = d.get('rf_gross', 1.0)
        best_f, max_util = self.compute_allocation(r_grid_spot, p_dens_ret, rf_g)
        self.ax_pdf.set_title(f"Return Dist {d['date']} | γ = {gamma:.2f}")
        self.ax_pdf.set_xlim(-1, 1)
        self.ax_pdf.grid(True)
        self.plot_fan()
        self.fig.canvas.draw_idle()
if __name__ == '__main__':
    TICKER = 'spy'
    yc = YieldCurve()
    print(f'\nAnalyzing {TICKER} with P-Splines...')
    tk_stock, spot_stock, hist_stock = get_data(TICKER)
    IS_INDEX_OBJ = TICKER in ['^SPX', 'SPY', '^NDX', '^GSPC', 'QQQ', 'IWM']
    tk_mkt, spot_mkt = (None, None)
    if not IS_INDEX_OBJ:
        print('Fetching ^SPX data for market variance reference...')
        try:
            tk_mkt, spot_mkt, _ = get_data('^SPX')
        except Exception as e:
            print(f'Could not fetch ^SPX data: {e}')
    if tk_stock:
        exps = tk_stock.options
        results = []
        now = datetime.now()
        valid_exps = []
        for e in exps:
            dt = datetime.strptime(e, '%Y-%m-%d')
            days = (dt - now).days
            if 5 < days < 800:
                valid_exps.append((e, days))
        if len(valid_exps) > 15:
            valid_exps = valid_exps[::2]
        print(f'Processing {len(valid_exps)} expirations...')
        for exp_str, days in valid_exps:
            try:
                print(f'  {exp_str}...', end='\r')
                T = days / 365.0
                r_dynamic = yc.get_rate(T)
                df_s = get_chain(tk_stock, exp_str)
                df_mkt = None
                if tk_mkt:
                    try:
                        df_mkt = get_chain(tk_mkt, exp_str)
                    except:
                        pass
                res = ExpectedReturnModel(r_dynamic, T).calculate(df_s, spot_stock, is_index=IS_INDEX_OBJ, df_market=df_mkt, market_spot=spot_mkt, manual_smoothing=1.0)
                res['date'] = exp_str
                res['chain_df'] = df_s
                res['market_chain_df'] = df_mkt
                res['market_spot'] = spot_mkt
                res['rf_used'] = r_dynamic
                results.append(res)
            except Exception as e:
                pass
        print('\nOptimization Complete.')
        avg_rf = yc.get_rate(1.0)
        if results and plt:
            dashboard = AnalysisDashboard(results, hist_stock, TICKER, avg_rf)
            plt.show()
