import threading
import queue
import socket
import time



def listener(q):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 43009))
        s.settimeout(10)
        s.listen(1)

        while True:
            try:
                conn, addr = s.accept()
                message = conn.recv(1024)
                data = message.decode('utf-8')
                q.put(data)
            except TimeoutError:
                pass
            except Exception:
                pass

if __name__ == '__main__':
    q = queue.Queue()
    listener_thread = threading.Thread(target=listener, args=(q,), daemon=True)

    try:
        listener_thread.start()
        while True:
            line = q.get()
            q.task_done()
            print(f">>>{line}>>>")

    except KeyboardInterrupt:
        pass
    except Exception as e:
        pass