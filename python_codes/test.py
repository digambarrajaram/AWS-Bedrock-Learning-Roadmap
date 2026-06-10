def update_config(filePath, Key, Value):
    with open(filePath, "r") as read:
        lines = read.readlines()
        for i in lines:
            print(i)

    with open (filePath, "w") as write:
        for line in lines:
            if Key in line:
                line = write.write(Key + "=" + Value + "\n")
            else:
               write.write(line)

update_config("server.config", "SERVER_IP", "172.2.0.5")

   