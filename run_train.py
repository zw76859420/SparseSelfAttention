from argparse import ArgumentParser
import subprocess
import os
def run_train():
    subprocess.call(['python', 'main.py', '--input_streams', "sub", '--prior', 'bernoulli', '--ber_prior', '0.25', '--lambda_val', '100.0', '--multihead', "4",
                     '--exp_name', 'finalfinal_SparseSA(wonorm,4head)_bernoulli_nonoise_0.25_lambda_100.0_tau1_sub_max_test'])

if __name__ == "__main__":
    parser = ArgumentParser(description='training code')
    parser.add_argument('-gpu', default="1", type=str)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    run_train()
