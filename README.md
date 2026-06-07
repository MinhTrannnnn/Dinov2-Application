# DINOv2 Product Inspect

Demo backend + frontend dung pretrained `dinov2_vits14` cho hai tac vu:

- Tim anh san pham tuong tu / phat hien anh trung gan giong.
- Phat hien bat thuong tren san pham bang patch features va heatmap.

## Cau truc data

Bo anh vao cac thu muc sau:

```text
data/
  gallery/   # anh dung de tim top-k anh giong nhat
  normal/    # anh san pham binh thuong, dung lam chuan anomaly
```

Data demo hien da duoc chuan bi:

```text
bottle:  83 gallery, 20 normal
tile:   117 gallery, 33 normal
leather:124 gallery, 32 normal
wood:    79 gallery, 19 normal
```

Goi y setup nhanh:

- `data/gallery`: co the chua ca anh normal va anh defect de demo truy hoi anh.
- `data/normal`: chi nen chua anh khong loi, cung loai san pham, chup tu goc nhin/crop gan nhau.
- Neu dung MVTec AD, chon 1 class truoc, vi du `bottle`, `tile`, `leather`, `wood`.
- Copy anh `train/good` vao `data/normal`.
- Copy anh `train/good` va mot so anh `test/*` vao `data/gallery`.
- Khi query, upload anh trong `test/good` hoac `test/defect_type`.

## Tai data demo tu dong

Khuyen nghi nhanh nhat: dung subset `bottle` tren Hugging Face DefectSpectrum. Tap nay nho hon archive MVTec day du, co anh `good` va 3 loai defect: `broken_large`, `broken_small`, `contamination`.

```powershell
python scripts\prepare_hf_defectspectrum.py --category bottle
```

Script se:

- Lay danh sach file tu Hugging Face.
- Tai anh vao `data/raw/defectspectrum/bottle`.
- Copy anh `good` vao `data/normal/bottle_good`.
- Copy anh `good` va defect vao `data/gallery/...`.

Neu muon dung MVTec AD archive day du, co script rieng:

```powershell
python scripts\prepare_mvtec.py --category bottle
```

Link host archive doi khi bi chan/het han; script Hugging Face o tren on dinh hon cho demo nhanh.

Co the doi category:

```powershell
python scripts\prepare_mvtec.py --category tile
python scripts\prepare_mvtec.py --category metal_nut
python scripts\prepare_mvtec.py --category zipper
```

## Chay backend

Lan dau chay backend se tai model `dinov2_vits14` tu PyTorch Hub. Can internet cho lan tai dau tien.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8010
```

API chinh:

- `GET /api/health`
- `GET /api/index/status`
- `POST /api/index/build`
- `POST /api/analyze`

## Chay frontend

```powershell
cd frontend
npm install
npm run dev
```

Mo `http://127.0.0.1:5173`.

## Workflow demo

1. Bo anh vao `data/gallery` va `data/normal`.
2. Start backend.
3. Start frontend.
4. Chon category, vi du `tile`.
5. Bam `Build index`.
6. Upload anh query va bam `Analyze`.
7. Xem top anh tuong tu, defect type du doan, verdict, closest normal, heatmap, defect box va PCA feature map.

## Chuc nang hien co

- Category selector: build va analyze rieng `tile`, `bottle`, khong tron index.
- Similar image search: tim top-k anh gan nhat trong gallery bang global embedding.
- Near duplicate: bao gan-trung neu similarity vuot nguong.
- Anomaly detection: so patch embedding voi anh good trong `data/normal`.
- PASS/WARNING/FAIL verdict: dua tren anomaly score va threshold.
- Likely defect type: tong hop nhan cua top similar images, vi du `crack`, `oil`, `rough`.
- Closest normal reference: anh good gan nhat de so sanh.
- Defect localization box: khoanh vung bat thuong tu heatmap.
- PCA feature map: chieu embedding 2D de xem cac cum good/defect.

## Ghi chu ve nguong

- Duplicate threshold mac dinh la `0.92`. Neu cung mot san pham nhung anh chup khac goc qua nhieu, ha xuong `0.85-0.90`.
- Anomaly threshold mac dinh la `0.35`. Nen test voi vai anh `good` truoc, sau do tang/giam de heatmap khop voi defect hon.
- Demo anomaly se dep nhat khi anh normal va anh query co bo cuc tuong doi dong nhat.

## Model

Backend dang load dung model:

```python
torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
```

`dinov2_vits14` nhe, phu hop demo laptop/CPU hon `vitb14`, nhung van du tot cho retrieval va heatmap anomaly co ban.
