import matplotlib.pyplot as plt

def display_statistics(stage,epochs,losses,acc):

    loss_x = [n for n in range(epochs)]
    loss_y = losses
    acc_x = [n for n in range(epochs)]
    acc_y = acc

    plt.subplot(1, 2, 1)
    plt.plot(loss_x, loss_y, 'o-')
    plt.title('%s loss statistics'%stage)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(acc_x, acc_y, '.-')
    plt.title('%s acc statistics'%stage)
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    # plt.show()
    plt.savefig('%s_statistics.jpg'%stage)


# import random
# n = 100
# list1 = [pow(i,3)+random.randint(0,i) for i in range(n)]
# display_statistics(n,list1,list1)
# while True:
#     pass
