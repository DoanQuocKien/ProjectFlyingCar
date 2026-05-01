import socket
import requests
import keyboard
import time

def find_car():
    # 1. Figure out what subnet the laptop is on right now
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        laptop_ip = s.getsockname()[0]
    except Exception:
        laptop_ip = '127.0.0.1'
    finally:
        s.close()

    subnet = '.'.join(laptop_ip.split('.')[:-1]) + '.'
    print(f"Scanning hotspot network ({subnet}X) for the car. This takes ~5 seconds...")

    # 2. Scan all 255 possible IP addresses on that subnet
    for i in range(1, 255):
        ip = f"{subnet}{i}"
        if ip == laptop_ip:
            continue

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.1) # Blazing fast timeout to skip dead IPs
        result = sock.connect_ex((ip, 80))
        sock.close()

        # 3. If an IP has a web server, test if it's our car
        if result == 0:
            try:
                url = f"http://{ip}/stop"
                response = requests.get(url, timeout=0.5)
                if response.status_code == 200:
                    print(f"\n[TARGET ACQUIRED] Car found at: {ip}\n")
                    return f"http://{ip}"
            except requests.exceptions.RequestException:
                pass

    print("\n[ERROR] Car not found. Is it turned on and connected to nth=)))) ?")
    return None

# --- Main Boot Sequence ---
#ESP32_IP = find_car()
ESP32_IP = "http://192.168.137.228"

if ESP32_IP:
    current_state = "stop"

    def send_command(cmd):
        global current_state
        if current_state != cmd:
            try:
                requests.get(f"{ESP32_IP}/{cmd}?speed=150", timeout=5)
                print(f"Command Sent: {cmd}")
                current_state = cmd
            except requests.exceptions.RequestException as e:
                print(f"Network error: {e}")

    print("4-Way Car Control Ready! Use Arrow Keys to drive. Press 'ESC' to quit.")

    # --- The Driving Loop ---
    while True:
        if keyboard.is_pressed('esc'):
            send_command("stop")
            print("Exiting...")
            break

        if keyboard.is_pressed('up'):
            send_command("forward")
        elif keyboard.is_pressed('down'):
            send_command("backward")
        elif keyboard.is_pressed('left'):
            send_command("left")
        elif keyboard.is_pressed('right'):
            send_command("right")
        else:
            send_command("stop")
            
        time.sleep(0.05)