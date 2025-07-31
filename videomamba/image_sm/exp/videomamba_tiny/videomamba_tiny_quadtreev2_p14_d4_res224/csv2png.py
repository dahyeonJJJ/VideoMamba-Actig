import pandas as pd
import matplotlib.pyplot as plt

# 1) CSV 읽어오기 ─── csv 파일이 다른 경로에 있으면 경로를 수정하세요.
df1 = pd.read_csv("log_out.csv")
df2 = pd.read_csv("/ephemeral/jdh/VideoMamba/videomamba/image_sm/exp/videomamba_tiny/videomamba_tiny_quadtreev2_nohibert_nosplit_p14_d4_res224/log_out.csv")

# 2) epoch vs test_acc1 그래프 그리기
plt.figure(figsize=(10, 6))

# 첫 번째 데이터 (파란색)
plt.plot(
    df1["epoch"],
    df1["test_acc1"],
    marker="o",
    markersize=0.5,
    linewidth=1,
    color="blue",
    label="quadtree_hibert_roisplit"
)

# 두 번째 데이터 (빨간색)
plt.plot(
    df2["epoch"],
    df2["test_acc1"],
    marker="o",
    markersize=0.5,
    linewidth=1,
    color="red",
    label="quadtree"
)

plt.title("Epoch vs. Test Accuracy (Top-1)")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy Top-1 (%)")
plt.grid(True)
plt.legend()

# 3) 그림 저장하기
output_path = "epoch_vs_test_acc1_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"그래프가 '{output_path}' 파일로 저장되었습니다!")
