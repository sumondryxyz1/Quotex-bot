# স্থায়ী লিংক (permanent link) বানানো

## আগে সত্যিটা জেনে নিন

এখন যে লিংকে ড্যাশবোর্ড চলছে:

```
https://work-1-ajdlpljacpvdqsuz.prod-runtime.all-hands.dev/
```

এটা **স্থায়ী নয়**। এর কারণ:

- এটা একটা অস্থায়ী runtime pod (`runtime-ajdlpljacpvdqsuz-...`)।
- লিংকের `ajdlpljacpvdqsuz` অংশটাই এই runtime-এর ID।
- runtime মুছে গেলে / নতুন করে তৈরি হলে লিংক **বদলে যাবে**।

তাই স্থায়ী লিংক পেতে ড্যাশবোর্ডকে একটা **স্থায়ী হোস্টে** চালাতে হবে। কোড ইতিমধ্যেই
তার জন্য প্রস্তুত (নিচে দেখুন)।

> ফিড বাইরের সার্ভারে যায় (`ws2.market-qx.info`), তাই যেকোনো হোস্ট থেকে চলবে।
> কেবল `data/candles.db` সেভ রাখতে একটা ডিস্ক/volume লাগবে।

---

## ⚠️ সবচেয়ে জরুরি কথা: এক IP = এক কানেকশন

পরীক্ষা করে দেখা গেছে — market-qx **একই IP থেকে একটাই লাইভ কানেকশন** দেয়।

- দুটো জায়গায় (হোস্ট + কন্টেইনার, বা দুই কন্টেইনার) একসাথে চালালে **দুটোই অন্ধ**
  হয়ে বসে থাকে (`assets=0`)।
- একটাই চালালে সাথে সাথে ৯২টি পেয়ার চলে আসে।

**অর্থাৎ:** নতুন হোস্টে deploy করার আগে **পুরোনোটা বন্ধ করুন**।

---

## অপশন A — সবচেয়ে সহজ, সত্যিকারের স্থায়ী লিংক (সুপারিশ)

Fly.io বা Render দিলে `https://<নাম>.fly.dev` টাইপের লিংক পাবেন, যা **প্রতিবার
redeploy করলেও একই থাকে**।

### যা লাগবে
- একটা বিনামূল্যের GitHub অ্যাকাউন্ট
- এই কোডটা একটা GitHub repo-তে push করা

### ধাপ ১ — GitHub-এ কোড তোলা

```bash
cd /workspace/project
git add -A && git commit -m "signal dashboard"
# আপনার GitHub-এ একটা খালি repo বানিয়ে তার URL দিন:
git remote add origin https://github.com/<আপনার-নাম>/<repo>.git
git push -u origin master
```

### ধাপ ২ — Fly.io-তে deploy

```bash
# একবারই ইনস্টল + লগইন
curl -L https://fly.io/install.sh | sh
fly auth login

cd /workspace/project
fly launch --no-deploy          # বিদ্যমান Dockerfile + fly.toml গ্রহণ করুন
fly volumes create candles --size 1   # ক্যান্ডেল ডেটার জন্য ডিস্ক
fly deploy
```

ব্যস। লিংক হবে `https://<app-name>.fly.dev` — এটাই আপনার স্থায়ী লিংক।

### Render-এ চাইলে
1. render.com → New → **Web Service** → আপনার GitHub repo বাছুন
2. Runtime: **Docker** (Dockerfile স্বয়ংক্রিয়ভাবে ধরা পড়বে)
3. Environment variable: `CANDLE_DB=/data/candles.db`
4. একটা **Disk** যোগ করুন, mount path `/data`, size 1GB
5. Deploy → `https://<নাম>.onrender.com` পাবেন

---

## অপশন B — নিজের VPS (DigitalOcean / Hetzner / Contabo)

মাসে ~$4-৫। সবচেয়ে বেশি নিয়ন্ত্রণ, লিংক আপনি নিজেই ঠিক করেন।

```bash
# VPS-এ (Docker ইনস্টল করা অবস্থায়)
git clone <আপনার-repo-url> qx && cd qx
docker build -t qx-dashboard .
docker run -d --name qx \
  -p 80:12000 \
  -e CANDLE_DB=/data/candles.db \
  -v qx-candles:/data \
  --restart unless-stopped \
  qx-dashboard
```

তারপর একটা ডোমেইন (`example.com`) VPS-এর IP-তে পয়েন্ট করুন। HTTPS-এর জন্য
[Caddy](https://caddyserver.com) সবচেয়ে সহজ — এক লাইনে সার্টিফিকেট হয়ে যায়।

---

## ডেটা (ক্যান্ডেল ইতিহাস) সেভ রাখা

ড্যাশবোর্ড নিজেই SQLite-এ ইতিহাস জমা করে `CANDLE_DB` পাথে। হোস্ট বদলালেও ইতিহাস
রাখতে ওই ফাইলটা **volume**-এ রাখুন (উপরের সব উদাহরণে `/data`)। এটাই আপনার
একমাত্র স্থায়ী সম্পদ — লিংক বদলালেও ডেটা হারাবে না।

---

## ডকার দিয়ে স্থানীয়ভাবে চালানো (পরীক্ষার জন্য)

```bash
./run_docker.sh          # পোর্ট 12000
./run_docker.sh 8080     # অন্য পোর্ট
docker logs -f qx-dashboard
```

লিংক স্থানীয় (`http://localhost:12000`) — এটা ইন্টারনেটে খোলা নয়, শুধু পরীক্ষার জন্য।

---

## সারসংক্ষেপ

| পদ্ধতি | লিংক স্থায়ী? | খরচ | কষ্ট |
| --- | --- | --- | --- |
| এখনকার work-1 লিংক | ❌ না | — | — |
| **Fly.io / Render** | ✅ হ্যাঁ | ফ্রি টিয়ার | কম |
| নিজের VPS | ✅ হ্যাঁ | ~$5/মাস | মাঝারি |

স্থায়ী লিংক পেতে **অবশ্যই** কোডটা একটা স্থায়ী হোস্টে (Fly/Render/VPS) চালাতে
হবে — শুধু এই runtime-এর লিংক কপি করে রাখলে সেটা একদিন মরে যাবে।
