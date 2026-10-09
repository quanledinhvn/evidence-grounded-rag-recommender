# Chạy CF trên AWS bằng SkyPilot

Cấu hình này làm theo cùng mô hình với project UAV:

- train trên disk local của EC2;
- archive artifact lên một S3 bucket do SkyPilot tự tạo hoặc tái sử dụng;
- mỗi lượt chạy có run ID UTC riêng;
- prefix kết quả có marker `SUCCESS` hoặc `FAILED`;
- EC2 tự down, còn bucket và artifact được giữ lại.

## Chuẩn bị

```bash
uv tool install 'skypilot[aws]==0.13.0'
aws configure
sky check aws
```

AWS principal cần quyền cho SkyPilot quản lý EC2 và S3. Không ghi access key,
secret key hoặc session token vào repository.

## Chạy

Chạy grid đầy đủ trong region Sydney:

```bash
./infra/skypilot/run.sh --region ap-southeast-2 --mode full
```

Full task tune đúng bốn cấu hình, là tích Descartes của `factors` trong
`{32, 64}` và `epochs` trong `{30, 60}`. Learning rate được giữ ở `0.01`,
regularization ở `0.001` và seed ở `42`. Tuning dùng một mẫu cố định 50.000
validation query. Model thắng mới được đánh giá trên toàn bộ validation và
test; BPR evaluation chạy theo batch 256 user. Popularity chỉ được so sánh ở
hai lượt đánh giá đầy đủ này.

Chạy smoke-train một epoch trên positive CSR đầy đủ, không chạy evaluator:

```bash
./infra/skypilot/run.sh --region ap-southeast-2 --mode check
```

`--region` là AWS region; SkyPilot tự chọn availability zone khả dụng trong
region đó. `--mode` chọn task `full` hoặc `check`. Nếu bỏ qua, giá trị mặc định
lần lượt là `ap-southeast-1` và `full`. Dùng `--help` để xem cú pháp.

Script có thể được gọi từ bất kỳ current directory nào. Nó tự tìm repository
root, lấy AWS account ID và suy ra tên bucket:

```text
hust-it4613-cf-<AWS_ACCOUNT_ID>-<AWS_REGION>
```

Không cần truyền `--env S3_BUCKET` và không cần tạo bucket trước. Task dùng
`name`, `store: s3` và `mode: MOUNT`, nên SkyPilot tạo bucket khi chưa có và
tái sử dụng bucket ở các lượt sau.

## Bố cục kết quả

Mỗi lượt tạo một run ID dạng `YYYYMMDDTHHMMSSZ`. Grid đầy đủ được archive tại:

```text
s3://<bucket>/runs/cf-movies-tv-2023-full/<run-id>/
```

Task kiểm tra được archive tại:

```text
s3://<bucket>/runs/cf-movies-tv-2023-check/<run-id>/
```

Task train vào `$HOME/cf-results/...` trên EC2 trước. Khi shell kết thúc, exit
trap sao chép kết quả hiện có sang S3. `SUCCESS` chỉ được tạo khi train và bước
sao chép đều thành công; nếu train thoát lỗi bình thường, prefix nhận marker
`FAILED` và giữ các artifact từng phần đã tạo được.

Spot interruption hoặc instance mất đột ngột có thể không cho exit trap chạy.
Vì pipeline chưa checkpoint/resume, cấu hình mặc định dùng On-Demand.

## Theo dõi và tải artifact

```bash
sky status
sky logs it4613-cf-full
```

Xác định bucket mặc định và liệt kê các lượt full:

```bash
aws_account_id=$(aws sts get-caller-identity --query Account --output text)
s3_bucket="hust-it4613-cf-${aws_account_id}-ap-southeast-2"

aws s3 ls "s3://$s3_bucket/runs/cf-movies-tv-2023-full/"
```

Tải một lượt cụ thể:

```bash
run_id=20261009T083015Z
mkdir -p "runs/cf-movies-tv-2023-full/$run_id"

aws s3 sync \
  "s3://$s3_bucket/runs/cf-movies-tv-2023-full/$run_id/" \
  "runs/cf-movies-tv-2023-full/$run_id/"
```

Một lượt full thành công gồm `evaluation.json`, `locked_config.json`,
`validation_metrics.csv`, bundle `validation/`, bundle `test/`, các candidate
trong `tuning/`, `train.log`, `run-metadata.txt` và marker `SUCCESS`.
`run-metadata.txt` cũng ghi phiên bản `implicit` thực tế đã khóa bởi `uv.lock`.

## Tùy chỉnh

Dùng bucket khác mà vẫn không cần sửa YAML:

```bash
S3_BUCKET=my-existing-bucket \
  ./infra/skypilot/run.sh --region ap-southeast-2 --mode full
```

Hai task YAML vẫn có thể launch thủ công, nhưng khi đó phải truyền cả
`S3_BUCKET` và `RUN_ID`. Wrapper là giao diện chạy được khuyến nghị.

Không cần GPU cho implementation hiện tại. BPR dùng backend Cython/OpenMP của
`implicit` trên CPU; `--num-threads 0` dùng các core khả dụng trong lúc fit một
candidate. Cả task full và check đều truyền tùy chọn này rõ ràng. Negative
sample verification giữ giá trị mặc định `true`; các candidate trong grid vẫn
được fit tuần tự.

Production training và evaluation được chạy bằng hai task SkyPilot trên AWS.
Local workflow chỉ dùng cho unit test nhỏ; không chạy dataset Movies and TV
đầy đủ trên máy local.

Setup cài `libgomp1`, OpenMP runtime mà wheel Linux của `implicit` cần. Khi
archive, task dùng recursive copy thay vì archive mode vì S3 mount không hỗ trợ
đầy đủ permission và timestamp POSIX.
