import asyncio
import time

async def run_print(text):

    #while(True):
    print(text)
    time.sleep(1)

async def run_wait_for_cmd():

    #while (True):
    cmd = input()

    if(cmd[2].isdigit()):
        if(cmd[0] == "S"): # SERVO COMMAND -- data count: 1 {0,1,2,3,4} - {up, down, left, right, default}
            print("Received command for Servo: " ,cmd[2])
        elif (cmd[0] == "M"):  # MOVEMENT COMMAND -- data count: 1 {0,1,2,3} - {forward, backward, left, right}
            print("Received command for Movement: ", cmd[2])
        elif (cmd[0] == "C"):  # SETTINGS COMMAND
            print("Received command for Setings: ", cmd[2])
        else:
            print("Wrong command format")
    else:
        print("Wrong command format")



async def main():
    while(True):
        await asyncio.gather(
            run_print("Test text"),
            run_wait_for_cmd()
        )

loop = asyncio.ProactorEventLoop()
asyncio.set_event_loop(loop)
loop.run_until_complete(main())