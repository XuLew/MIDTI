import torch
import codecs

from code.pretrain_code.ReGCN import ReGCNs
from code.pretrain_code.torch_data import outputCSVfile
from code.pretrain_code.dataprocessing import data_pre
from code.pretrain_code.param import parameter_parser
# from code.torch_data import outputCSVfile

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def list_write_csv(file_name, l): 
    f = codecs.open(file_name, 'w')  
    for line in l:
        f.write(str(line) + '\n')
    f.close()


def pretrain(model, train_data, optimizer, opt):

    # model.train()
    epoch_list = []
    loss_list = []
    for epoch in range(0, opt.epoch):
        print('epoch',epoch)
        model.zero_grad()
        feature, matrix = model(train_data)
        # loss =torch.nn.BCEWithLogitsLoss(reduction='mean')
        loss = torch.nn.MSELoss(reduction='mean') 
        if opt.name == 'drug':
            loss1 = loss(matrix, train_data['dd_dr']['data_matrix'].to(device))
            loss2 = loss(matrix, train_data['dd_di']['data_matrix'].to(device))
            loss3 = loss(matrix, train_data['dd_se']['data_matrix'].to(device))
            loss4 = loss(matrix, train_data['dd_str']['data_matrix'].to(device))
            loss5 = loss(matrix, train_data['dd_pro']['data_matrix'].to(device))
            loss = loss1 + loss2 + loss3 + loss4 + loss5
        else:
            loss1 = loss(matrix, train_data['pp_pro']['data_matrix'].to(device))
            loss2 = loss(matrix, train_data['pp_di']['data_matrix'].to(device))
            loss3 = loss(matrix, train_data['pp_seq']['data_matrix'].to(device))
            loss4 = loss(matrix, train_data['pp_dr']['data_matrix'].to(device))
            loss = loss1 + loss2 + loss3 + loss4
        epoch_list.append(epoch)
        loss_list.append(loss.item())

        loss.backward()
        optimizer.step()
        print(epoch, loss.item())
        print(feature)
        print(matrix)


    scoremin, scoremax = matrix.min(), matrix.max()
    matrix = (matrix - scoremin) / (scoremax - scoremin) 

    feature = feature.cpu().detach().numpy()
    matrix = matrix.cpu().detach().numpy()
    # np.savetxt(opt.name + '_feature' + '.txt', feature, fmt='%.10f', delimiter=' ')
    # np.savetxt(opt.name + '_matrix' + '.txt', matrix, fmt='%.10f', delimiter=' ')

    if opt.name == 'drug':
        # dim or dim_without_VA
        outputCSVfile(opt.train_dataset_path + 'dim/' + opt.name + '_f{}_feature{}.csv'.format(opt.f, opt.d_out_channels), feature)
        outputCSVfile(opt.train_dataset_path + 'dim/' + opt.name + '_f{}_matrix{}.csv'.format(opt.f, opt.d_out_channels), matrix)
        list_write_csv(opt.train_dataset_path + 'dim/' + opt.name + '_f{}_recon{}_losslist.csv'.format(opt.f, opt.d_out_channels), loss_list)
    else:
        outputCSVfile(opt.train_dataset_path + 'dim/' + opt.name + '_f{}_feature{}.csv'.format(opt.f, opt.p_out_channels), feature)
        outputCSVfile(opt.train_dataset_path + 'dim/' + opt.name + '_f{}_matrix{}.csv'.format(opt.f, opt.p_out_channels), matrix)
        list_write_csv(opt.train_dataset_path + 'dim' + opt.name + '_f{}_recon{}_losslist.csv'.format(opt.f, opt.p_out_channels), loss_list)

    # plt.plot(epoch_list, loss_list)
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.savefig(opt.train_dataset_path + 'dim/' + opt.name + '_f{}_recon{}_loss.png'.format(opt.f, opt.out_channels))
    # plt.show()

    return feature, matrix

def main(name):
    args = parameter_parser(name)
    pretrain_data = data_pre(args)
    torch.cuda.empty_cache()

    model = ReGCNs(args)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    feature, matrix = pretrain(model, pretrain_data, optimizer, args)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main("drug")
    main("protein")



