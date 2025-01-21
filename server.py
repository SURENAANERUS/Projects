import socket
import threading


SERVER_ADDRESS = '127.0.0.1'
SERVER_PORT = 8000

# Please put your code in this file

# HTTP 1.1 communication takes place over TCP/IP connections
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((SERVER_ADDRESS, SERVER_PORT))

server.listen()

# Descriptions and cat_urls lists will store data submitted via form
# They have two empty entries at start in order to prevent out of bounds errors
descriptions = ["", ""]
cat_urls = ["", ""]

print(f"Serving HTTP on port {SERVER_PORT}")

# The function starts new threads for new connections. In persistent connection there should be at most 2 at a time
def run_server():
    while True:
        client, address = server.accept()
        print (f"Connected to a client at address: {address}")
        threading.Thread(target=handle_thread, args=(client,), daemon=True).start()

# This function is the heart of the application. It parses the requests and sends responses using helper functions
# Args: client (socket.socket)
def handle_thread(client):
    # Try block catches errors raised by get_message() when the connection terminates
    try:
        while True:
            # HTTP requests begin with a request line
            # It consists of the method, request URI, and HTTP version
            request_line = get_message(client)
            fields = request_line.split(" ")
            method = fields[0]
            print(f"\nRequest Line: {request_line}")

            # Check for GET method
            if method == "GET":
                print("a GET request was received")
                # path from the request line
                path = fields[1]

                # Receive the many header fields
                # End of this sequence is indicated by an empty message
                while True:
                    header_field = get_message(client)
                    if len(header_field) == 1:
                        break
                
                # The main page may be requested using two alternative paths
                if path == "/" or path == "/index.html":
                    print("The main page was requested")
                    # Call a helper function to get file content
                    file_content = get_content("data/index.html")
                    # Call a helper function to send a response
                    # Code: 200, description: "OK"
                    send_response(file_content, client, 200, "OK")
                
                # "/img" at the beginning of the path indicates request for an image
                elif path.startswith("/img"):
                    print("An image was requested")
                    # Select file name from the path
                    file_name = path[5:]
                    file_path = f"data/img/{file_name}"
                    
                    # Read raw bytes
                    with open(file_path, "rb") as file:
                        file_content = file.read()

                    # Send response. Code: 200, description: "OK"
                    send_response(file_content, client, 200, "OK")
                
                # Request for personal cats page
                elif path == "/personal_cats.html":
                    with open("data/personal_cats.html", "r") as file:
                        cont = file.read()
                    
                    # This may look vague, but thanks to this trick, it is possible to replace the respective fragments
                    # of the html file with data
                    cont = cont.replace("Cat name 1", descriptions[-2])
                    cont = cont.replace("Cat name 2", descriptions[-1])
                    cont = cont.replace("Here should be your source url", cat_urls[-1])
                    cont = cont.replace(cat_urls[-1], cat_urls[-2], 1)
                    # Encode using UTF-8
                    cont = cont.encode("utf-8")                                   
                    # Send a response
                    send_response(cont, client, 200, "OK")                   

                # All other cases
                else:
                    # Attempt to read an arbitrary file. If the path is incorrected, an excoption will occur.
                    try:
                        # Send the file on success
                        file_content = get_content(path[1:])
                        send_response(file_content, client, 200, "OK")
                    
                    # Exception occurred
                    except:
                        print("Not found")
                        # Load 404.html page and send code 404
                        file_content = get_content("data/404.html")
                        send_response(file_content, client, 404, "Not Found")    

            # Check for the POST method
            if method == "POST":
                # Receive the header fields
                while True:
                    header_field = get_message(client)
                    if len(header_field) == 1:
                        break
                    # One of the headers will contain the content-length in bytes. We need it to correctly receive the body
                    header, value = header_field.split(": ")
                    if header == "Content-Length":
                        content_length = int(value)
                
                # Receive the body using the content length
                body = client.recv(content_length).decode("utf-8")
                # Call the helper function url_decode() to decode the body which is in the form of "application/x-www-form-urlencoded"
                # The function returns the value of cat_url and description
                cat_url, description = url_decode(body)

                # If one of the fields is empty, respond with bad request
                if not cat_url or not description:
                    print("Bad Request")
                    # Send 400.html page
                    content = get_content("data/400.html")
                    send_response(content, client, 400, "Bad Request")

                # If not, append data to the lists and respond with code 201 - "Created"
                else:
                    cat_urls.append(cat_url)
                    descriptions.append(description)
                    # Send the success.html page
                    success_content = get_content("data/success.html")
                    send_response(success_content, client, 201, "Created")

    # Excpetion in the case of conncetion termination
    except Exception as e:
        print(type(e).__name__)
        print(e)
        client.close()

# Function that sends an HTTP response, a sequence of header fields and body
# Args: file_content (str), client (socket.socket), code (int), description (str)
def send_response(file_content, client, code, description):
    file_length = len(file_content)
    # Send header fields
    client.send(f"HTTP/1.1 {code} {description}\r\n".encode("utf-8"))
    client.send(f"Content-Length: {file_length}\r\n".encode("utf-8"))
    client.send("Content-Type: text/html; charset=UTF-8\r\n".encode("utf-8"))
    client.send("Connection: Open\r\n".encode("utf-8"))
    # Seperate header fields from the body with an empty message
    client.send("\r\n".encode("utf-8"))
    # Send the body using the helper function send_long_mes()
    send_long_mes(client, file_content)

# Function for opening the file and reading its content
# Args: file_path (str)
# Returns: Encoded content
def get_content(file_path):
    with open (file_path, "r") as file:
        file_content = file.read()
    return file_content.encode("utf-8")

# Function for sending messages. Same was used in previous assignments
def send_long_mes(dest_client, message):
    bytes_to_send = len(message)
    bytes_sent = dest_client.send(message)

    if bytes_sent <= 0:
        raise ConnectionError()

    while bytes_sent != bytes_to_send:
        bytes_sent += dest_client.send(message[bytes_sent:])

# Function for receiving the message. Same was used in previous assignments
def get_message(client):
    message = ""
    while True:
        char = client.recv(1).decode("utf-8")
        if not char:
            print("Socket disconnected")
            raise ConnectionError()
        if char == "\n":
            return message
        message += char

# Function for decoding the "application/x-www-form-urlencoded" format
# Args: encoding (str)
# Returns: cat_url (str), description (str)
def url_decode(encoding):
    # Pairs of name and value are seperatad with the ampersand
    fields = encoding.split("&")
    # Obtain the values
    _, description = fields[0].split("=")
    _, cat_url = fields[1].split("=")
    # Helper function for special chars called when returning
    return percent_decode(cat_url), percent_decode(description)

# In this encoding, special chars are encoded in the form %XX where XX is a 2-digit hex number, representing the ascii value
# This function, decodes such characters
def percent_decode(encoding):
    decoded = ""
    i = 0
    while (i < len(encoding)):
        char = encoding[i]
        if char == "%": 
            # char_to_int() helper function is used
            ascii_value = char_to_int(encoding[i + 1]) * 16
            ascii_value += char_to_int(encoding[i + 2])
            decoded += chr(ascii_value)
            i += 2
        elif char == "+":
            decoded += " "
        else:
            decoded += char
        i += 1
    return decoded

# Takes hex digits in the form of chars and returns an integer value
def char_to_int(char):
    val = ord(char)
    if ord("a") <= val and val <= ord("f"):
        integer = 10 + val - ord("a")
    elif ord("A") <= val and val <= ord("F"):
        integer = 10 + val - ord("A")
    else:
        integer = val - ord("0")
    return integer

# Function to terminate the server with keyboard input. Solely for convenience
def close_server():
    while True:
        command = input()
        if command == "!quit":
            server.close()
            break

# The main run_server thread is started as well as a thread for server termination
threading.Thread(target=run_server, daemon=True).start()
threading.Thread(target=close_server).start()