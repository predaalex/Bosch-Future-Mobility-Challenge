def subscribeToConfig(queuesList, message, receiver, pipeSend):
    queuesList["Config"].put(
        {
            "Subscribe/Unsubscribe": "subscribe",
            "Owner": message.Owner.value,
            "msgID": message.msgID.value,
            "To": {
                "receiver": receiver,
                "pipe": pipeSend,
            },
        }
    )

def sendMessageToQueue(queuesList, message, value):
    queuesList[message.Queue.value].put(
        {
            "Owner": message.Owner.value,
            "msgID": message.msgID.value,
            "msgType": message.msgType.value,
            "msgValue": value,
        }
    )
