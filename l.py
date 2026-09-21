import urllib3
import requests
import numpy as np
import matplotlib.pyplot as plt

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_URL = (
    "https://api.open-elevation.com/api/v1/lookup?locations="
    "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|"
    "48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|"
    "48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|"
    "48.166497,24.523574|48.166128,24.520214|48.165416,24.517176|"
    "48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|"
    "48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|"
    "48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
)

response = requests.get(API_URL, verify=False)
raw_points = response.json()["results"]
num_nodes = len(raw_points)

print(f"Дані успішно завантажено з API. Кількість вузлів: {num_nodes}")

def calculate_haversine(lat1, lon1, lat2, lon2):
    EARTH_RADIUS = 6371000.0 
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    d_lat = np.radians(lat2 - lat1)
    d_lon = np.radians(lon2 - lon1)
    
    val_a = np.sin(d_lat / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lon / 2.0)**2
    return 2.0 * EARTH_RADIUS * np.arctan2(np.sqrt(val_a), np.sqrt(1.0 - val_a))

gps_coords = [(pt["latitude"], pt["longitude"]) for pt in raw_points]
altitudes = np.array([pt["elevation"] for pt in raw_points])

cum_distances = [0.0]
for idx in range(1, num_nodes):
    dist_step = calculate_haversine(
        gps_coords[idx-1][0], gps_coords[idx-1][1],
        gps_coords[idx][0], gps_coords[idx][1]
    )
    cum_distances.append(cum_distances[-1] + dist_step)

cum_distances = np.array(cum_distances)

with open("tabulation_results.txt", "w", encoding="utf-8") as out_file:
    out_file.write("№ | Distance (m) | Elevation (m)\n")
    out_file.write("-" * 35 + "\n")
    for idx in range(num_nodes):
        out_file.write(f"{idx:2d} | {cum_distances[idx]:10.2f} | {altitudes[idx]:8.2f}\n")

print("Результати табуляції успішно збережено у файл 'tabulation_results.txt'.")


def compute_spline_params(x_arr, y_arr):
    n_intervals = len(x_arr) - 1
    step_h = np.diff(x_arr)
    
    vec_alpha = np.zeros(n_intervals)
    for i in range(1, n_intervals):
        vec_alpha[i] = (3.0 / step_h[i]) * (y_arr[i+1] - y_arr[i]) - (3.0 / step_h[i-1]) * (y_arr[i] - y_arr[i-1])
        
    diag_l = np.ones(n_intervals + 1)
    coeff_mu = np.zeros(n_intervals + 1)
    vec_z = np.zeros(n_intervals + 1)
    
    for i in range(1, n_intervals):
        diag_l[i] = 2.0 * (x_arr[i+1] - x_arr[i-1]) - step_h[i-1] * coeff_mu[i-1]
        coeff_mu[i] = step_h[i] / diag_l[i]
        vec_z[i] = (vec_alpha[i] - step_h[i-1] * vec_z[i-1]) / diag_l[i]
        
    coeff_c = np.zeros(n_intervals + 1)
    coeff_b = np.zeros(n_intervals)
    coeff_d = np.zeros(n_intervals)
    coeff_a = y_arr[:-1]
 
    for j in range(n_intervals - 1, -1, -1):
        coeff_c[j] = vec_z[j] - coeff_mu[j] * coeff_c[j+1]
        coeff_b[j] = (y_arr[j+1] - y_arr[j]) / step_h[j] - step_h[j] * (coeff_c[j+1] + 2.0 * coeff_c[j]) / 3.0
        coeff_d[j] = (coeff_c[j+1] - coeff_c[j]) / (3.0 * step_h[j])
        
    return coeff_a, coeff_b, coeff_c[:-1], coeff_d

def get_spline_points(x_grid, x_nodes, a, b, c, d):
    y_res = np.zeros_like(x_grid)
    for pos, point in enumerate(x_grid):
        segment_idx = np.searchsorted(x_nodes, point) - 1
        segment_idx = np.clip(segment_idx, 0, len(x_nodes) - 2)
        delta_x = point - x_nodes[segment_idx]
        y_res[pos] = (a[segment_idx] + 
                      b[segment_idx] * delta_x + 
                      c[segment_idx] * (delta_x**2) + 
                      d[segment_idx] * (delta_x**3))
    return y_res

a_vec, b_vec, c_vec, d_vec = compute_spline_params(cum_distances, altitudes)

print("\nКоефіцієнти кубічних сплайнів (a, b, c, d):")
for i in range(len(a_vec)):
    print(f"Сплайн {i:2d}: a={a_vec[i]:.2f}, b={b_vec[i]:.4f}, c={c_vec[i]:.4f}, d={d_vec[i]:.6f}")

plt.figure(figsize=(11, 5))
plt.plot(cum_distances, altitudes, "ko", label="Вихідні точки (GPS)")

test_nodes = [8, 12, 16]
colors = ["#d95f02", "#7570b3", "#1b9e77"]

for idx, count in enumerate(test_nodes):
    sel_indices = np.linspace(0, num_nodes - 1, count, dtype=int)
    sub_dist = cum_distances[sel_indices]
    sub_alt = altitudes[sel_indices]
    sa, sb, sc, sd = compute_spline_params(sub_dist, sub_alt)
    x_line = np.linspace(sub_dist[0], sub_dist[-1], 350)
    y_line = get_spline_points(x_line, sub_dist, sa, sb, sc, sd)
    plt.plot(x_line, y_line, color=colors[idx], linewidth=1.8, label=f"Сплайн ({count} вузлів)")

plt.title("Порівняння кубічних сплайнів залежно від кількості вузлів")
plt.xlabel("Кумулятивна відстань (м)")
plt.ylabel("Висота (м)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

x_fine_grid = np.linspace(cum_distances[0], cum_distances[-1], 600)
y_spline_approx = get_spline_points(x_fine_grid, cum_distances, a_vec, b_vec, c_vec, d_vec)
y_linear = np.interp(x_fine_grid, cum_distances, altitudes)
abs_error = np.abs(y_linear - y_spline_approx)

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

ax_top.plot(x_fine_grid, y_linear, "r--", label=r"Задана функція $y=f(x)$ (ламана)", alpha=0.7)
ax_top.plot(x_fine_grid, y_spline_approx, "b-", label="Наближене значення (Кубічний сплайн)")
ax_top.plot(cum_distances, altitudes, "ko", label="Вузлові точки")
ax_top.set_ylabel("Висота (м)")
ax_top.set_title("Порівняння заданої функції та її наближеного значення кубічним сплайном")
ax_top.legend()
ax_top.grid(True, linestyle="--")

ax_bot.plot(x_fine_grid, abs_error, color="darkmagenta", label=r"Похибка $\epsilon = |y - y_{набл}|$")
ax_bot.set_xlabel("Кумулятивна відстань на відрізку [x_0, x_n] (м)")
ax_bot.set_ylabel("Похибка (м)")
ax_bot.set_title("Графік похибки інтерполяції")
ax_bot.legend()
ax_bot.grid(True, linestyle="--")

plt.tight_layout()
plt.show()

print("\nХарактеристики маршруту:")
total_dist = cum_distances[-1]
climb_up = sum(max(altitudes[i] - altitudes[i-1], 0) for i in range(1, num_nodes))
climb_down = sum(max(altitudes[i-1] - altitudes[i], 0) for i in range(1, num_nodes))

x_eval = np.linspace(cum_distances[0], cum_distances[-1], 600)
y_eval = get_spline_points(x_eval, cum_distances, a_vec, b_vec, c_vec, d_vec)
slopes = np.gradient(y_eval, x_eval) * 100.0

tourist_mass = 60.0  
gravity = 9.81
work_joules = tourist_mass * gravity * climb_up

print(f"Загальна довжина маршруту (м): {total_dist:.2f}")
print(f"Сумарний набір висоти (м): {climb_up:.2f}")
print(f"Сумарний спуск (м): {climb_down:.2f}")
print(f"Максимальний підйом (%): {np.max(slopes):.2f}")
print(f"Максимальний спуск (%): {np.min(slopes):.2f}")
print(f"Середній градієнт (%): {np.mean(np.abs(slopes)):.2f}")
print(f"Механічна робота (Дж): {work_joules:.2f}")
print(f"Механічна робота (кДж): {work_joules / 1000.0:.2f}")
print(f"Енергія (ккал): {work_joules / 4184.0:.2f}")