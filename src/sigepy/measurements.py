import time
import board
import busio
import csv
import os
import numpy as np
from datetime import datetime
from gpiozero import LED
from adafruit_ads1x15.analog_in import AnalogIn
import adafruit_ads1x15.ads1115 as ADS
from mpu6050 import mpu6050
import matplotlib.pyplot as plt
from collections import deque
from scipy.fft import fft, fftfreq
import threading
import queue


# Variables para graficar
ventana_muestras = 528  # Numero de puntos que se mostraron en el grafico
elongaciones = deque([0] * ventana_muestras, maxlen=ventana_muestras)
elongaciones2 = deque([0] * ventana_muestras, maxlen=ventana_muestras)  # Segunda medicion de desplazamiento
aceleraciones1_x = deque([0] * ventana_muestras, maxlen=ventana_muestras)
aceleraciones1_y = deque([0] * ventana_muestras, maxlen=ventana_muestras)
aceleraciones1_z = deque([0] * ventana_muestras, maxlen=ventana_muestras)
aceleraciones2_x = deque([0] * ventana_muestras, maxlen=ventana_muestras)
aceleraciones2_y = deque([0] * ventana_muestras, maxlen=ventana_muestras)
aceleraciones2_z = deque([0] * ventana_muestras, maxlen=ventana_muestras)
diferencias_x = deque([0] * ventana_muestras, maxlen=ventana_muestras)
diferencias_y = deque([0] * ventana_muestras, maxlen=ventana_muestras)
diferencias_z = deque([0] * ventana_muestras, maxlen=ventana_muestras)


plt.ion()  # Modo interactivo
plt.close('all')  # Cerrar todas las figuras anteriores
plt.ioff()  # Desactivar modo interactivo al inicio

fig = plt.figure(figsize=(10, 12))
ax = [fig.add_subplot(4, 1, i + 1) for i in range(4)]

fig_fft = plt.figure(figsize=(10, 10))
ax_fft = [fig_fft.add_subplot(3, 1, i + 1) for i in range(3)]
plt.figure(fig.number)  # Activar la figura principal para evitar ventanas emergentes inesperadas

# Variables para registro de sismos
en_registro_sismo = False
tiempo_ultimo_trigger = 0
buffer_pre_trigger = deque([], maxlen=int(PRE_TRIGGER_TIEMPO / delta_t))
datos_evento_actual = []
queue_eventos = queue.Queue()





def actualizar_graficos(accel1, accel2, elongacion1, elongacion2, diferencias):
    """Actualiza las graficas de aceleraciones y desplazamiento."""
    global fig, ax

    # Actualizar datos
    elongaciones.append(elongacion1)
    elongaciones2.append(elongacion2)
    aceleraciones1_x.append(accel1["x"])
    aceleraciones1_y.append(accel1["y"])
    aceleraciones1_z.append(accel1["z"])
    aceleraciones2_x.append(accel2["x"])
    aceleraciones2_y.append(accel2["y"])
    aceleraciones2_z.append(accel2["z"])
    diferencias_x.append(diferencias["x"])
    diferencias_y.append(diferencias["y"])
    diferencias_z.append(diferencias["z"])

    # Limpiar ejes
    for i in range(4):
        ax[i].clear()

    # Graficar elongaciones
    ax[0].plot(range(len(elongaciones)), list(elongaciones), label="Desp 1", color='b')
    ax[0].plot(range(len(elongaciones2)), list(elongaciones2), label="Desp 2", color='g')
    ax[0].set_title("Desplazamientos")
    ax[0].set_ylabel("Desp (mm)")
    ax[0].legend()
    if len(elongaciones) > 0 and len(elongaciones2) > 0:
        ax[0].set_ylim([min(min(elongaciones), min(elongaciones2)) - 1, max(max(elongaciones), max(elongaciones2)) + 1])

    # Graficar aceleraciones en X
    ax[1].plot(range(len(aceleraciones1_x)), list(aceleraciones1_x), label="Accel1 X", color='r')
    ax[1].plot(range(len(aceleraciones2_x)), list(aceleraciones2_x), label="Accel2 X", color='g')
    ax[1].set_title("Aceleraciones en X")
    ax[1].set_ylabel("Accel (m/s2)")
    ax[1].legend()

    # Graficar aceleraciones en Y
    ax[2].plot(range(len(aceleraciones1_y)), list(aceleraciones1_y), label="Accel1 Y", color='orange')
    ax[2].plot(range(len(aceleraciones2_y)), list(aceleraciones2_y), label="Accel2 Y", color='b')
    ax[2].set_title("Aceleraciones en Y")
    ax[2].set_ylabel("Accel (m/s2)")
    ax[2].legend()

    # Graficar aceleraciones en Z
    ax[3].plot(range(len(aceleraciones1_z)), list(aceleraciones1_z), label="Accel1 Z", color='purple')
    ax[3].plot(range(len(aceleraciones2_z)), list(aceleraciones2_z), label="Accel2 Z", color='yellow')
    ax[3].set_title("Aceleraciones en Z")
    ax[3].set_ylabel("Accel (m/s2)")
    ax[3].legend()

    # Redibujar sin bloquear
    fig.canvas.draw()
    fig.canvas.flush_events()


def calcular_y_graficar_fft():
    """Calcula y grafica la FFT para los ejes X, Y, Z del acelerometro 1."""
    global fig_fft, ax_fft

    # Limpiar ejes FFT
    for i in range(3):
        ax_fft[i].clear()

    # Solo calcular FFT si tenemos suficientes puntos
    if len(aceleraciones1_x) >= ventana_muestras:
        # Calcular FFT para cada eje
        datos_x = np.array(list(aceleraciones1_x))
        datos_y = np.array(list(aceleraciones1_y))
        datos_z = np.array(list(aceleraciones1_z))

        # Aplicar ventana Hanning para reducir fugas espectrales
        window = np.hanning(len(datos_x))
        datos_x = datos_x * window
        datos_y = datos_y * window
        datos_z = datos_z * window

        # Calcular FFT
        fft_x = np.abs(fft(datos_x))
        fft_y = np.abs(fft(datos_y))
        fft_z = np.abs(fft(datos_z))

        # Frecuencias correspondientes
        freqs = fftfreq(len(datos_x), delta_t)

        # Solo mostrar la primera mitad (frecuencias positivas)
        n = len(freqs) // 2

        # Graficar FFT eje X
        ax_fft[0].plot(freqs[:n], fft_x[:n], 'r')
        ax_fft[0].set_title("FFT Aceleracion X")
        ax_fft[0].set_xlabel("Frecuencia (Hz)")
        ax_fft[0].set_ylabel("Amplitud")
        ax_fft[0].grid(True)

        # Graficar FFT eje Y
        ax_fft[1].plot(freqs[:n], fft_y[:n], 'g')
        ax_fft[1].set_title("FFT Aceleracion Y")
        ax_fft[1].set_xlabel("Frecuencia (Hz)")
        ax_fft[1].set_ylabel("Amplitud")
        ax_fft[1].grid(True)

        # Graficar FFT eje Z
        ax_fft[2].plot(freqs[:n], fft_z[:n], 'b')
        ax_fft[2].set_title("FFT Aceleracion Z")
        ax_fft[2].set_xlabel("Frecuencia (Hz)")
        ax_fft[2].set_ylabel("Amplitud")
        ax_fft[2].grid(True)

    fig_fft.tight_layout()
    fig_fft.canvas.draw()
    fig_fft.canvas.flush_events()


def detectar_sismo(accel1, accel2, timestamp):
    """
    Detecta y registra eventos sismicos basados en umbrales de aceleracion.
    Implementa un sistema de trigger/detrigger con buffer pre-evento.
    """
    global en_registro_sismo, tiempo_ultimo_trigger, datos_evento_actual

    # Obtener la magnitud de aceleracion combinada (vector resultante)
    magnitud1 = np.sqrt(accel1['x'] ** 2 + accel1['y'] ** 2 + accel1['z'] ** 2)

    # Determinar si se supera el umbral de trigger
    es_trigger = magnitud1 > TRIGGER_UMBRAL

    # Almacenar datos en el buffer pre-trigger
    datos_actuales = {
        'timestamp': timestamp,
        'accel1': accel1.copy(),
        'accel2': accel2.copy(),
        'magnitud': magnitud1
    }
    buffer_pre_trigger.append(datos_actuales)

    # Logica de deteccion de sismo
    tiempo_actual = time.time()

    if es_trigger:
        tiempo_ultimo_trigger = tiempo_actual

        if not en_registro_sismo:
            # Inicio de un nuevo evento sismico
            print(f"ALERTA! Posible evento sismico detectado: {timestamp}")
            en_registro_sismo = True

            # Guardar buffer pre-trigger
            datos_evento_actual = list(buffer_pre_trigger)

        # Anadir datos actuales al evento
        datos_evento_actual.append(datos_actuales)

    elif en_registro_sismo:
        # Anadir datos durante el periodo post-trigger
        datos_evento_actual.append(datos_actuales)

        # Verificar si ha pasado suficiente tiempo desde el ultimo trigger
        if tiempo_actual - tiempo_ultimo_trigger > POST_TRIGGER_TIEMPO:
            # Finalizar registro del evento
            print(f"Fin de evento sismico: {timestamp}")

            # Verificar duracion minima
            if len(datos_evento_actual) * delta_t >= TIEMPO_MINIMO_SISMO:
                # Anadir evento a la cola para procesamiento
                queue_eventos.put(datos_evento_actual)
                print(f"Evento valido registrado - duracion: {len(datos_evento_actual) * delta_t:.2f} segundos")
            else:
                print(f"Evento descartado - duracion insuficiente: {len(datos_evento_actual) * delta_t:.2f} segundos")

            # Reiniciar para proximo evento
            en_registro_sismo = False
            datos_evento_actual = []


def procesar_eventos():
    """
    Funcion para procesar eventos sismicos en segundo plano.
    Se ejecuta en un hilo separado para no interferir con la adquisicion de datos.
    """
    while True:
        try:
            # Esperar a que haya un evento en la cola
            datos_evento = queue_eventos.get(timeout=1)

            # Generar nombre de archivo con timestamp
            timestamp_inicio = datos_evento[0]['timestamp']
            timestamp_str = timestamp_inicio.strftime('%Y%m%d_%H%M%S')
            nombre_archivo = f"{DIRECTORIO_EVENTOS}/sismo_{timestamp_str}.csv"

            # Guardar datos del evento
            with open(nombre_archivo, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    "Timestamp",
                    "Accel1_X", "Accel1_Y", "Accel1_Z",
                    "Accel2_X", "Accel2_Y", "Accel2_Z",
                    "Magnitud"
                ])

                for dato in datos_evento:
                    writer.writerow([
                        dato['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f'),
                        f"{dato['accel1']['x']:.4f}",
                        f"{dato['accel1']['y']:.4f}",
                        f"{dato['accel1']['z']:.4f}",
                        f"{dato['accel2']['x']:.4f}",
                        f"{dato['accel2']['y']:.4f}",
                        f"{dato['accel2']['z']:.4f}",
                        f"{dato['magnitud']:.4f}"
                    ])

            # Tambien guardar en formato binario (mas eficiente)
            np_datos = {
                'timestamps': np.array([d['timestamp'].timestamp() for d in datos_evento]),
                'accel1_x': np.array([d['accel1']['x'] for d in datos_evento]),
                'accel1_y': np.array([d['accel1']['y'] for d in datos_evento]),
                'accel1_z': np.array([d['accel1']['z'] for d in datos_evento]),
                'accel2_x': np.array([d['accel2']['x'] for d in datos_evento]),
                'accel2_y': np.array([d['accel2']['y'] for d in datos_evento]),
                'accel2_z': np.array([d['accel2']['z'] for d in datos_evento]),
                'magnitud': np.array([d['magnitud'] for d in datos_evento])
            }
            np.savez(f"{DIRECTORIO_EVENTOS}/sismo_{timestamp_str}.npz", **np_datos)

            # Generar un reporte de analisis basico
            generar_reporte_sismo(nombre_archivo, np_datos, timestamp_str)

            print(f"Evento sismico guardado: {nombre_archivo}")
            queue_eventos.task_done()

        except queue.Empty:
            # No hay eventos en la cola, continuar esperando
            continue
        except Exception as e:
            print(f"Error procesando evento sismico: {e}")


def generar_reporte_sismo(archivo_csv, datos_np, timestamp_str):
    """
    Genera un reporte basico del evento sismico con analisis estadistico
    y espectral.
    """
    # Calcular estadisticas
    max_accel_x = np.max(np.abs(datos_np['accel1_x']))
    max_accel_y = np.max(np.abs(datos_np['accel1_y']))
    max_accel_z = np.max(np.abs(datos_np['accel1_z']))
    max_magnitud = np.max(datos_np['magnitud'])
    duracion = len(datos_np['timestamps']) * delta_t

    # Calcular FFT para analisis frecuencial
    fft_x = np.abs(fft(datos_np['accel1_x']))
    fft_y = np.abs(fft(datos_np['accel1_y']))
    fft_z = np.abs(fft(datos_np['accel1_z']))
    freqs = fftfreq(len(datos_np['accel1_x']), delta_t)

    # Encontrar frecuencias dominantes (primeras 3)
    n = len(freqs) // 2  # Solo frecuencias positivas
    idx_x = np.argsort(fft_x[:n])[-3:][::-1]
    idx_y = np.argsort(fft_y[:n])[-3:][::-1]
    idx_z = np.argsort(fft_z[:n])[-3:][::-1]

    frec_dom_x = freqs[idx_x]
    frec_dom_y = freqs[idx_y]
    frec_dom_z = freqs[idx_z]

    # Generar imagenes del evento
    plt.figure(figsize=(12, 10))

    # Grafico de aceleraciones
    plt.subplot(2, 1, 1)
    t = np.arange(0, duracion, delta_t)[:len(datos_np['accel1_x'])]
    plt.plot(t, datos_np['accel1_x'], 'r', label='X')
    plt.plot(t, datos_np['accel1_y'], 'g', label='Y')
    plt.plot(t, datos_np['accel1_z'], 'b', label='Z')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Aceleracion (m/s2)')
    plt.title(f'Evento Sismico - {timestamp_str}')
    plt.grid(True)
    plt.legend()

    # Grafico de FFT
    plt.subplot(2, 1, 2)
    plt.plot(freqs[:n], fft_x[:n], 'r', label='X')
    plt.plot(freqs[:n], fft_y[:n], 'g', label='Y')
    plt.plot(freqs[:n], fft_z[:n], 'b', label='Z')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud')
    plt.title('Analisis Espectral')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{DIRECTORIO_EVENTOS}/sismo_{timestamp_str}_analisis.png")
    plt.close()

    # Generar reporte de texto
    with open(f"{DIRECTORIO_EVENTOS}/sismo_{timestamp_str}_reporte.txt", 'w') as f:
        f.write(f"REPORTE DE EVENTO SISMICO\n")
        f.write(f"=========================\n\n")
        f.write(f"Fecha y hora: {datetime.fromtimestamp(datos_np['timestamps'][0])}\n")
        f.write(f"Duracion: {duracion:.2f} segundos\n\n")
        f.write(f"ACELERACIONES MAXIMAS:\n")
        f.write(f"  Eje X: {max_accel_x:.4f} m/s2\n")
        f.write(f"  Eje Y: {max_accel_y:.4f} m/s2\n")
        f.write(f"  Eje Z: {max_accel_z:.4f} m/s2\n")
        f.write(f"  Magnitud resultante: {max_magnitud:.4f} m/s2\n\n")
        f.write(f"ANALISIS FRECUENCIAL:\n")
        f.write(f"  Frecuencias dominantes X: {', '.join([f'{f:.2f} Hz' for f in frec_dom_x])}\n")
        f.write(f"  Frecuencias dominantes Y: {', '.join([f'{f:.2f} Hz' for f in frec_dom_y])}\n")
        f.write(f"  Frecuencias dominantes Z: {', '.join([f'{f:.2f} Hz' for f in frec_dom_z])}\n\n")
        f.write(f"Archivos relacionados:\n")
        f.write(f"  - {archivo_csv}\n")
        f.write(f"  - {DIRECTORIO_EVENTOS}/sismo_{timestamp_str}.npz\n")
        f.write(f"  - {DIRECTORIO_EVENTOS}/sismo_{timestamp_str}_analisis.png\n")


# Iniciar hilo para procesar eventos sismicos
hilo_procesamiento = threading.Thread(target=procesar_eventos, daemon=True)
hilo_procesamiento.start()

# Contador para actualizacion de FFT (no necesitamos calcularla en cada iteracion)
contador_fft = 0
INTERVALO_FFT = 10  # Actualizar FFT cada 10 iteraciones
# Bucle principal
if __name__ == "__main__":
    try:
        print("Iniciando sistema de monitoreo sismico...")
        print(f"Frecuencia de muestreo: {SAMPLE_RATE} Hz")
        print(f"Umbral de trigger: {TRIGGER_UMBRAL} m/s2")
        print(f"Directorio de eventos: {DIRECTORIO_EVENTOS}")
        print("Presione Ctrl+C para finalizar")

        print(f"Tamano de la cola: {queue_eventos.qsize()}")

        # Mostrar las figuras iniciales
        fig.show()
        fig_fft.show()
        while True:
            # Obtener timestamp actual
            timestamp_actual = datetime.now()

            # Leer datos del ADS1115
            v0_1, elongacion1, v0_2, elongacion2 = lectura_ads1115()

            # Leer datos de los acelerometros
            accel1 = mpu1.get_accel_data()
            accel2 = mpu2.get_accel_data()

            # Aplicar offsets
            for eje in offsets_mpu1:
                accel1[eje] -= offsets_mpu1[eje]
                accel2[eje] -= offsets_mpu2[eje]

            # Calcular diferencias
            diferencias = {eje: accel1[eje] - accel2[eje] for eje in ["x", "y", "z"]}

            # Encender o apagar el LED segun el umbral de desplazamiento
            if elongacion1 > threshold:
                led_18.on()  # Enciende el LED si la elongacion supera el umbral
            else:
                led_18.off()  # Apaga el LED si la elongacion es menor que el umbral

            # Imprimir datos en consola
            impresion_datos(v0_1, elongacion1, v0_2, elongacion2, accel1, accel2, diferencias)

            # Actualizar graficos
            actualizar_graficos(accel1, accel2, elongacion1, elongacion2, diferencias)

            # Actualizar FFT cada cierto numero de iteraciones
            contador_fft += 1
            if contador_fft >= INTERVALO_FFT:
                calcular_y_graficar_fft()
                contador_fft = 0

            # Detectar y registrar eventos sismicos
            detectar_sismo(accel1, accel2, timestamp_actual)

            # Guardar datos en el archivo CSV principal de mediciones
            guardar_csv(csv_file,
                        [timestamp_actual.strftime('%Y-%m-%d %H:%M:%S.%f'), f"{v0_1:.4f}", f"{elongacion1:.2f}",
                         f"{elongacion2:.2f}", f"{accel1['x']:.2f}", f"{accel1['y']:.2f}", f"{accel1['z']:.2f}",
                         f"{accel2['x']:.2f}", f"{accel2['y']:.2f}", f"{accel2['z']:.2f}",
                         f"{diferencias['x']:.2f}", f"{diferencias['y']:.2f}", f"{diferencias['z']:.2f}"])

            # Guardar datos en el archivo CSV para desplazamientos si se supera el umbral de desplazamientos
            if elongacion1 > umbral_desplazamiento or elongacion2 > umbral_desplazamiento:
                guardar_csv(csv_file_desplazamiento, [timestamp_actual.strftime('%Y-%m-%d %H:%M:%S.%f'), f"{v0_1:.4f}",
                                                      f"{elongacion1:.2f}", f"{elongacion2:.2f}"])

            # Guardar datos en el archivo CSV para aceleraciones si se superan los umbrales criticos de aceleracion
            if abs(accel1['x']) > umbral_aceleracion or abs(accel1['y']) > umbral_aceleracion or \
                    abs(accel1['z']) > umbral_aceleracion or abs(accel2['x']) > umbral_aceleracion or \
                    abs(accel2['y']) > umbral_aceleracion or abs(accel2['z']) > umbral_aceleracion:
                guardar_csv(csv_file_aceleracion, [timestamp_actual.strftime('%Y-%m-%d %H:%M:%S.%f'),
                                                   f"{accel1['x']:.2f}", f"{accel1['y']:.2f}", f"{accel1['z']:.2f}",
                                                   f"{accel2['x']:.2f}", f"{accel2['y']:.2f}", f"{accel2['z']:.2f}"])

            # Esperar para mantener el intervalo de muestreo constante
            time.sleep(delta_t)

    except KeyboardInterrupt:
        print("Programa detenido por el usuario.")
    except Exception as e:
        print(f"Error en el programa: {e}")
    finally:
        print("Finalizando el sistema...")
        # Apagar los LEDs
        led_17.off()
        led_18.off()

        # Cerrar correctamente las figuras
        plt.close('all')

        # Asegurarse de que el hilo de procesamiento termine
        if 'hilo_procesamiento' in locals() and hilo_procesamiento.is_alive():
            try:
                queue_eventos.put(None)  # Senal para terminar el hilo
                hilo_procesamiento.join(timeout=2)
            except:
                pass

        print("Sistema de monitoreo sismico finalizado.")
