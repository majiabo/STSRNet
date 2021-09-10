score = """
avg_ssim:[0.9648571  0.96056813 0.9546616  0.94652975 0.95716625]
avg_psnr:[36.176605 34.9679   34.192085 34.434574 36.706017]
avg_mse:[19.758297 24.449917 28.060476 26.388985 16.016668]
avg_mae:[2.1860154 2.4243543 2.5196128 2.6151798 2.2435038]
avg_lpips:[0.02383827 0.02470299 0.02567451 0.03479318 0.03107617]
"""
if __name__ == '__main__':
    lines = score.strip().split('\n')
    for line in lines:
        title, values = line.split(':')
        values = [float(i) for i in values[1:-1].split() if i]
        print(title, '{:.4f}'.format(sum(values)/len(values)))
