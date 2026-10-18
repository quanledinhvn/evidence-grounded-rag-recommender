# Smoke test 10 record — hệ thống cuối (fusion + bằng chứng + giải thích có trích dẫn)

Lệnh: `python -m backend.generation.experiment --phase test --sample 5 --with-target 5 --output-dir runs/llm/smoke-10 --report docs/smoke_test_10_record.md`; dữ liệu thô (gói bằng chứng, câu trả lời LLM, nhãn giám khảo) ở `runs/llm/smoke-10/`. Sinh tự động bởi `python -m backend.generation.experiment` (2026-10-18 21:57). Phần "Đánh giá tổng quan" ở cuối do người chạy viết.

## 1. Dữ liệu và cấu hình

| Thành phần | Giá trị |
|---|---|
| Pha | `test`: model fit trên train + validation; target là tương tác cuối của user (test) |
| Query | `artifacts\movies-tv-2023\queries_test.parquet` — mỗi user một target, `as_of` = thời điểm target |
| Mẫu | 5 query ngẫu nhiên (`Random(42)`) + 5 query có target trong top 10 (lấy đầu tiên trong 5000 query ngẫu nhiên; tỉ lệ có target trong top 10 của lượt quét: 10,2%) |
| Lịch sử, hồ sơ | `splits.parquet`, chỉ tương tác trước `as_of`; catalog `catalog_enriched.parquet`; phiên bản `editions.parquet` |
| Xếp hạng | Fusion `runs/fusion-editions-movies-tv-2023/test`: trọng số bm25 0.1, itemknn 1, popularity 1, method score, depth 200, gom phiên bản; giữ nguyên thứ tự fusion (không xếp lại) |
| Bằng chứng | Metadata catalog + tối đa 2 review của người khác viết trước `as_of` (`reviews.sqlite`) cho mỗi phim top 10 |
| LLM giải thích | `gemini-3.5-flash-lite`, temperature 0, JSON schema; top 3, tối đa 2 câu/phim; 2 chế độ: tự do và bắt buộc trích dẫn |
| Validator | ID nguồn tồn tại và đúng phim, H# thuộc hồ sơ, tên người/tựa trong câu có trong nguồn, độ giống MiniLM ≥ 0.3 |
| Giám khảo | `gemini-3.5-flash` (thinking low), chấm từng câu: supported / partial / unsupported so với nguồn của phim và lịch sử |

## 2. Chi tiết từng record

### Record 1 (ngẫu nhiên) — user `AFUCCFTAHYYQOQ6HUBPZIE5D7P4Q`

- **Query:** `as_of` = 1672686439044 (2023-01-02); target `B07Y9BWCGJ` — *The Peanut Butter Falcon*, rating 3★.
- **Kết quả:** target không có trong top 10.

**Bước 1 — Hồ sơ.** 20 phim đã xem trước `as_of`; 18 phim thích (≥ 4★), 0 phim không thích (≤ 2★); 0 phim không có metadata. Bằng chứng lịch sử đưa cho LLM:

- [H1] 5★ Groundhog Day (Special 15th Anniversary Edition) Genre: Romance, Comedy Actors: Bill Murray, Andie MacDowell, … — review của user: "Very enjoyable movie. Still a good movie even after all these years."
- [H2] 5★ The Best Exotic Marigold Hotel Actors: Judi Dench, Bill Nighy, Penelope Wilton, Dev Patel, … — review của user: "A very good story with excellent acting. We have watched it several times."
- [H3] 5★ Rick Steves Europe:12 New Shows DVD 2015–2016 Genre: TRAVEL, Europe, General Studio: Rick Steves …
- [H4] 5★ Last Action Hero Genre: Action & Adventure Actors: Arnold Schwarzenegger, Austin O'Brien, F. Murray …
- [H5] 5★ Guardians of the Galaxy Genre: Action/Adventure Actors: Chris Pratt, Zoe Saldana, Dave Bautista, Vin …
- [H6] 5★ Galaxy Quest [Deluxe Edition] Genre: Science Fiction & Fantasy, Comedy, Action & Adventure Actors: …
- … và 12 mục khác

**Bước 2 — Fusion top 10** (nhánh tìm ra phim, điểm fusion):

| Hạng | Phim | Nhánh | Điểm |
|---:|---|---|---:|
| 1 | It Happened on 5th Avenue (DVD) | itemknn | 1.984 |
| 2 | Holiday Inn | itemknn | 1.981 |
| 3 | The Hundred-Foot Journey | itemknn | 1.962 |
| 4 | Woman In Gold | itemknn | 1.855 |
| 5 | Remember the Night | itemknn | 1.852 |
| 6 | Miracle on 34th Street | itemknn | 1.711 |
| 7 | The Bishop's Wife | itemknn | 1.704 |
| 8 | Ladies in Lavender | itemknn | 1.691 |
| 9 | The Shop Around the Corner [DVD] | itemknn | 1.648 |
| 10 | Quartet | itemknn | 1.634 |

**Bước 3 — Bằng chứng.** 58 mục cho top 10 (38 metadata, 20 review). Của top 3:

- *It Happened on 5th Avenue (DVD)*: [E1] title: It Happened on 5th Avenue (DVD); [E2] genres: Holidays & Seasonal, Christmas, Comedy; [E3] description: It Happened on 5th Avenue (DVD) As he does every winter, …; [E4] Directors: Roy Del Ruth. Cast: Don DeFore, Ann Harding, Charles Ruggles, Victor …; [E5] When I first read the description of the movie in the TV …; [E6] This movie is a little silly at times, sometimes a little stiff, …
- *Holiday Inn*: [E7] title: Holiday Inn; [E8] genres: Holidays & Seasonal, Christmas, Comedy, Fred Astaire, Musicals & Performing Arts, …; [E9] I hadn't watched the movies "Holiday Inn" or "White Christmas" for years. …; [E10] I just finished watching the newly released Special Edition of Holiday Inn. …
- *The Hundred-Foot Journey*: [E11] title: The Hundred-Foot Journey; [E12] genres: Award Winners in Movies & TV, Golden Globe Central, Golden Globe …; [E13] description: Starring Academy Award(R) winner Helen Mirren (Best Actress, THE QUEEN, 2006), …; [E14] Directors: Lasse Hallstrom, Lasse Hallström. Cast: Helen Mirren, Om Puri, Manish Dayal, …; [E15] The Hundred-foot Journey has a hidden message, not picked up by the …; [E16] One of my current favorite movies! I just love how this movie …

**Bước 4 — LLM giải thích có trích dẫn** (2073 token vào, 263 token ra, cache; trạng thái ok) và **bước 5 — validator, giám khảo**:

| Phim | Câu LLM viết | Nguồn | Validator | Độ giống | Giám khảo |
|---|---|---|---|---:|---|
| It Happened on 5th Avenue (DVD) | It is a comedy with a Christmas setting, aligning with the user's enjoyment of holiday films like Elf. | E2 E3 H10 | giữ | 0.62 | supported |
| Holiday Inn | It features a Christmas and comedy genre mix similar to the holiday movies the user has loved. | E8 H10 | giữ | 0.61 | supported |
| The Hundred-Foot Journey | It is a romantic comedy, fitting the user's positive history with romance and comedy films. | E12 H1 H11 | giữ | 0.53 | supported |

**Lời giải thích hiển thị cho user** (chỉ các câu validator giữ lại):

- *It Happened on 5th Avenue (DVD)*: It is a comedy with a Christmas setting, aligning with the user's enjoyment of holiday films like Elf [E2][E3][H10].
- *Holiday Inn*: It features a Christmas and comedy genre mix similar to the holiday movies the user has loved [E8][H10].
- *The Hundred-Foot Journey*: It is a romantic comedy, fitting the user's positive history with romance and comedy films [E12][H1][H11].

**Đối chứng — giải thích tự do** (không bắt buộc trích dẫn, cùng thông tin):

- *It Happened on 5th Avenue (DVD)*: It features a heartwarming holiday comedy style that aligns with your love for feel-good movies like Elf. *(supported)* The charming story of unlikely companions sharing a home fits your taste for uplifting comedies. *(supported)*
- *Holiday Inn*: This classic holiday and comedy title matches your enjoyment of delightful, family-friendly seasonal films. *(supported)* It offers lighthearted entertainment similar to your highly rated comedy and musical preferences. *(partial)*
- *The Hundred-Foot Journey*: This uplifting romantic comedy shares the excellent ensemble acting and warm storytelling you praised in The Best Exotic Marigold Hotel. *(supported)* The blend of heart, humor, and strong character dynamics will appeal to your taste for engaging drama and comedy. *(supported)*

### Record 2 (ngẫu nhiên) — user `AHKI5BG7FIOMMJZJCGDVEO7GJN3Q`

- **Query:** `as_of` = 1495661692000 (2017-05-24); target `B00287Z18Y` — *Three Amigos (Repackage/DVD)*, rating 5★.
- **Kết quả:** target không có trong top 10.

**Bước 1 — Hồ sơ.** 33 phim đã xem trước `as_of`; 20 phim thích (≥ 4★), 1 phim không thích (≤ 2★); 0 phim không có metadata. Bằng chứng lịch sử đưa cho LLM:

- [H1] 5★ Heavy Metal [Blu-ray] Genre: Sci-Fi, Fantasy, Musical Director: Gerald Potterton Producers: Ivan Reitman Studio: …
- [H2] 5★ Akira: Movie (Blu-ray/DVD Combo) Genre: Anime & Manga Actors: Johnny Yong Bosch, Jan Rabson, …
- [H3] 5★ Ice Age 4-Movie Set Studio: 20th Century Fox Release date: April 1, 2014 Run … — review của user: "I only really like the first move but my kid absolutely loves them"
- [H4] 4★ A Bug's Life (Two-Disc Collector's Edition) Genre: Animation Actors: Diller, Phyllis, Foley, Dave, Garrett, … — review của user: "Not my favorite Pixar but my kid loves it that is why it gets 4 stars instead of 3"
- [H5] 4★ Thirteen Ghosts / House of Wax [Blu-ray] Actors: Various Director: Various Studio: WarnerBrothers Release …
- [H6] 5★ Wanderlust Genre: Comedy Actors: Jennifer Aniston, Paul Rudd Director: David Wain Studio: Universal Pictures … — review của user: "I almost died laughing when I watched this movie it is a must see"
- … và 15 mục khác

**Bước 2 — Fusion top 10** (nhánh tìm ra phim, điểm fusion):

| Hạng | Phim | Nhánh | Điểm |
|---:|---|---|---:|
| 1 | Krull | itemknn | 1.449 |
| 2 | Fire and Ice | itemknn | 1.442 |
| 3 | ICE AGE: COLLISION COURSE | itemknn | 1.393 |
| 4 | Ghost in the Shell: 25th Anniversary Edition [Blu-ray] | itemknn | 1.356 |
| 5 | Cowboy Bebop: The Complete Series [Blu-ray] | itemknn | 1.313 |
| 6 | Wizards (35th Anniversary Edition) [Blu-ray Book] | itemknn | 1.296 |
| 7 | Antz [DVD] | itemknn | 1.289 |
| 8 | Sin City [Blu-ray] | itemknn | 1.276 |
| 9 | Paprika [Blu-ray] | itemknn | 1.264 |
| 10 | Vampire Hunter D [Blu-ray] | itemknn | 1.236 |

**Bước 3 — Bằng chứng.** 55 mục cho top 10 (35 metadata, 20 review). Của top 3:

- *Krull*: [E1] title: Krull; [E2] genres: Science Fiction; [E3] description: Product Description Journey into a mystical time and place that belongs …; [E4] Directors: Peter Yates. Cast: Ken Marshall, Lysette Anthony, Liam Neeson, Freddie Jones, …; [E5] 1983’s sci-fi fantasy adventure “Krull” gets its Blu-ray debut thanks to Mill …; [E6] Krull is an incredibly fun little adventure movie. The music is great …
- *Fire and Ice*: [E7] title: Fire and Ice; [E8] description: Product Description From The Director Of The Original Animated THE LORD …; [E9] Directors: Ralph Bakshi. Cast: Randy Norton; [E10] What you have here is swords and sorcery cel-animated movie with all …; [E11] Is it possible to love something and hate something at the same …
- *ICE AGE: COLLISION COURSE*: [E12] title: ICE AGE: COLLISION COURSE; [E13] genres: Kids & Family; [E14] This is the first Ice Age film I did not like. I …; [E15] The fifth film in the animated Ice-Age series, ICE AGE: COLLISION COURSE …

**Bước 4 — LLM giải thích có trích dẫn** (2091 token vào, 440 token ra, cache; trạng thái ok) và **bước 5 — validator, giám khảo**:

| Phim | Câu LLM viết | Nguồn | Validator | Độ giống | Giám khảo |
|---|---|---|---|---:|---|
| Krull | You may enjoy Krull because you previously rated Heavy Metal, a sci-fi and fantasy film, with 5 out of 5 stars. | E2 E3 H1 | giữ | 0.52 | supported |
| Krull | A reviewer notes that the film features great music and an exciting mix of action and adventure. | E6 | giữ | 0.39 | supported |
| Fire and Ice | Fire and Ice is a fantasy and animation film directed by Ralph Bakshi, appealing to fans of animated adventures. | E8 E9 E10 | giữ | 0.41 | supported |
| ICE AGE: COLLISION COURSE | You may like this entry in the Ice Age franchise since you previously rated the Ice Age 4-Movie Set 5 out of 5 stars. | E12 E15 H3 | giữ | 0.60 | supported |
| ICE AGE: COLLISION COURSE | A reviewer mentions that this fifth installment in the animated series is a special treat for children and families. | E15 | độ giống thấp | 0.26 | supported |

**Lời giải thích hiển thị cho user** (chỉ các câu validator giữ lại):

- *Krull*: You may enjoy Krull because you previously rated Heavy Metal, a sci-fi and fantasy film, with 5 out of 5 stars [E2][E3][H1]. A reviewer notes that the film features great music and an exciting mix of action and adventure [E6].
- *Fire and Ice*: Fire and Ice is a fantasy and animation film directed by Ralph Bakshi, appealing to fans of animated adventures [E8][E9][E10].
- *ICE AGE: COLLISION COURSE*: You may like this entry in the Ice Age franchise since you previously rated the Ice Age 4-Movie Set 5 out of 5 stars [E12][E15][H3].

**Đối chứng — giải thích tự do** (không bắt buộc trích dẫn, cùng thông tin):

- *Krull*: You enjoyed sci-fi and fantasy films like Heavy Metal. *(supported)* This cult classic adventure blends mythical elements and great action. *(supported)*
- *Fire and Ice*: You appreciate animated masterpieces like Akira. *(partial)* This swords and sorcery film features unique animation style and fantasy collaboration. *(supported)*
- *ICE AGE: COLLISION COURSE*: You gave a high rating to the Ice Age 4-Movie Set. *(supported)* This fifth installment continues the animated family-friendly adventure series. *(supported)*

### Record 3 (ngẫu nhiên) — user `AEOUHTOKRS2MPWLN7FD44PMELLHA`

- **Query:** `as_of` = 1488463339000 (2017-03-02); target `B002IFD63I` — *The Jackal*, rating 1★.
- **Kết quả:** target không có trong top 10.

**Bước 1 — Hồ sơ.** 4 phim đã xem trước `as_of`; 1 phim thích (≥ 4★), 3 phim không thích (≤ 2★); 0 phim không có metadata. Bằng chứng lịch sử đưa cho LLM:

- [H1] 4★ Cast Away (Two-Disc Special Edition) Starring: Tom Hanks Helen Hunt Paul Sanchez Directors: Robert … — review của user: "entertaining I like the big engine in the water shot"
- [H2] 1★ Eastern Promises (Widescreen Edition) Starring: Viggo Mortensen Naomi Watts Vincent Cassel Directors: David Cronenberg …
- [H3] 1★ The Book of Eli Starring: Denzel Washington Gary Oldman Mila Kunis Directors: Allen Hughes … — review của user: "extremely boring...I rarely bail out of a movie...but I did for this one."
- [H4] 2★ Jack Ryan: Shadow Recruit Starring: Chris Pine Kevin Costner Kenneth Branagh Directors: Kenneth Branagh … — review của user: "good action, low on character development, clever use of dubbed Russian for the star bad guy."

**Bước 2 — Fusion top 10** (nhánh tìm ra phim, điểm fusion):

| Hạng | Phim | Nhánh | Điểm |
|---:|---|---|---:|
| 1 | The People Vs. Larry Flynt | itemknn | 1.304 |
| 2 | Detachment | itemknn | 1.270 |
| 3 | Ikiru | itemknn | 1.215 |
| 4 | Made in Britain | itemknn | 1.116 |
| 5 | The Raven | itemknn | 1.044 |
| 6 | The Polar Express (Two-Disc Widescreen Edition) | bm25 | 1.040 |
| 7 | Margin Call | itemknn | 1.034 |
| 8 | Directors: Sergio Navarretta. Cast: Louis Gossett Jr., Ana Golja, Shohreh Aghdashloo | itemknn | 1.033 |
| 9 | Captain Phillips | bm25 | 1.029 |
| 10 | The Kids Are All Right | itemknn | 1.027 |

**Bước 3 — Bằng chứng.** 37 mục cho top 10 (25 metadata, 12 review). Của top 3:

- *The People Vs. Larry Flynt*: [E1] title: The People Vs. Larry Flynt; [E2] Directors: Milos Forman. Cast: Woody Harrelson, Courtney Love, Edward Norton; [E3] This is one of the best First Ammendment films ever made. It's …; [E4] Wonderful acting done by lead performers.Want to watch again and again.Loved the …
- *Detachment*: [E5] title: Detachment; [E6] genres: Drama, Arthouse, Cerebral, Downbeat; [E7] description: A Tribeca Film. Adrien Brody (in his most acclaimed performance since …; [E8] Directors: Austin Stark. Cast: Adrien Brody, Tony Kaye
- *Ikiru*: [E9] title: Ikiru; [E10] genres: Drama, Emotional, Thoughtful; [E11] description: A dying man devotes his last months to building a children's …; [E12] Directors: Akira Kurosawa. Cast: Takashi Shimura, Nobuo Kaneko, Kyoko Seki; [E13] What would you do if you found out you only had a …; [E14] Simply beautiful. Magnificent cinematography and great acting. It really makes you contemplate …

**Bước 4 — LLM giải thích có trích dẫn** (1002 token vào, 257 token ra, cache; trạng thái ok) và **bước 5 — validator, giám khảo**:

| Phim | Câu LLM viết | Nguồn | Validator | Độ giống | Giám khảo |
|---|---|---|---|---:|---|
| The People Vs. Larry Flynt | The People Vs. Larry Flynt features wonderful acting by the lead performers according to viewer reviews. | E4 | giữ | 0.63 | supported |
| Detachment | Detachment stars Adrien Brody in an acclaimed drama about troubled education. | E7 | giữ | 0.56 | supported |
| Ikiru | Ikiru is described as an emotional and thoughtful drama about a dying man. | E10 E11 | giữ | 0.37 | supported |

**Lời giải thích hiển thị cho user** (chỉ các câu validator giữ lại):

- *The People Vs. Larry Flynt*: The People Vs. Larry Flynt features wonderful acting by the lead performers according to viewer reviews [E4].
- *Detachment*: Detachment stars Adrien Brody in an acclaimed drama about troubled education [E7].
- *Ikiru*: Ikiru is described as an emotional and thoughtful drama about a dying man [E10][E11].

**Đối chứng — giải thích tự do** (không bắt buộc trích dẫn, cùng thông tin):

- *The People Vs. Larry Flynt*: You may appreciate the strong dramatic performances and engaging courtroom elements. *(supported)* It offers a compelling story driven by talented lead actors. *(supported)*
- *Detachment*: This cerebral drama features acclaimed lead performances similar to intense character-driven films. *(supported)* It explores complex social themes in a troubled school environment. *(supported)*
- *Ikiru*: You might enjoy this thoughtful and emotional drama about a man making a meaningful impact. *(supported)* It features magnificent cinematography and causes viewers to deeply contemplate life. *(supported)*

### Record 4 (ngẫu nhiên) — user `AFM7KHOM6Z2SUZGMCFYMX64M5Z6A`

- **Query:** `as_of` = 1543637682793 (2018-12-01); target `B005LAIHPE` — *Battleship*, rating 3★.
- **Kết quả:** target không có trong top 10.

**Bước 1 — Hồ sơ.** 7 phim đã xem trước `as_of`; 7 phim thích (≥ 4★), 0 phim không thích (≤ 2★); 0 phim không có metadata. Bằng chứng lịch sử đưa cho LLM:

- [H1] 5★ O Brother, Where Art Thou? Genre: Comedy Actors: George Clooney, John Turturro, Tim Blake …
- [H2] 5★ Sling Blade [DVD + Digital] Genre: Drama Actors: Billy Bob Thornton, Dwight Yoakam, J.T. …
- [H3] 5★ Forrest Gump Genre: Drama, Comedy Actors: Robin Wright Penn, Sam Anderson, Sally Field, Gary …
- [H4] 5★ A Christmas Story (DVD) Actors: Peter Billingsley, Melinda Dillon, Darren McGavin, Ian Petrella, Scott …
- [H5] 5★ A Charlie Brown Christmas 50th Anniversay Deluxe Edition (DVD) Genre: Kids & Family Actors: …
- [H6] 5★ The Big Bang Theory: The Complete Tenth Season [DVD] Genre: Comedy Actors: Johnny Galecki, …
- … và 1 mục khác

**Bước 2 — Fusion top 10** (nhánh tìm ra phim, điểm fusion):

| Hạng | Phim | Nhánh | Điểm |
|---:|---|---|---:|
| 1 | Cast Away (Two-Disc Special Edition) | itemknn | 1.497 |
| 2 | Fargo: Special Edition | itemknn | 1.489 |
| 3 | National Lampoon's Christmas Vacation (Special Edition) [DVD] | itemknn | 1.465 |
| 4 | The Big Bang Theory: The Complete Eleventh Season (DVD) | itemknn | 1.461 |
| 5 | Unbreakable (Two-Disc Vista Series) | itemknn | 1.461 |
| 6 | Snatch | itemknn | 1.403 |
| 7 | The Big Bang Theory: Season 9 [DVD] | itemknn | 1.390 |
| 8 | Training Day | itemknn | 1.380 |
| 9 | Crouching Tiger, Hidden Dragon | itemknn | 1.379 |
| 10 | Catch Me If You Can (Widescreen Two-Disc Special Edition) | itemknn | 1.358 |

**Bước 3 — Bằng chứng.** 53 mục cho top 10 (33 metadata, 20 review). Của top 3:

- *Cast Away (Two-Disc Special Edition)*: [E1] title: Cast Away (Two-Disc Special Edition); [E2] genres: Drama; [E3] description: A contemporary drama about a man (Hanks) who is isolated and …; [E4] Directors: Robert Zemeckis. Cast: Tom Hanks, Helen Hunt, Paul Sanchez, Lari White, …; [E5] Ever since 1992s "A League of Their Own", Tom Hanks has been …; [E6] When I first saw this movie in theatres I was somewhat surprised …
- *Fargo: Special Edition*: [E7] title: Fargo: Special Edition; [E8] genres: Special Editions, Mystery & Suspense; [E9] description: Product Description Fargo: SE (RPKG/DVD) ]]> Amazon.com Leave it to the …; [E10] I give this movie 4 stars and a half, or 8,5/10 points. …; [E11] Fargo is a truly great modern-classic. Frances McDormand is touching with her …
- *National Lampoon's Christmas Vacation (Special Edition) [DVD]*: [E12] title: National Lampoon's Christmas Vacation (Special Edition) [DVD]; [E13] genres: Holidays & Seasonal, Christmas, Comedy; [E14] description: National Lampoon's Christmas Vacation: Special Edition (DVD) Make merry as Chevy …; [E15] Directors: Jeremiah Chechik. Cast: Chevy Chase, Beverly D'Angelo, Randy Quaid, Miriam Flynn, …; [E16] i have a certain weakness for the so-stupid-it's-funny school of comedy, but …; [E17] Yes people. I am this years Santa and if you don't think …

**Bước 4 — LLM giải thích có trích dẫn** (1659 token vào, 378 token ra, cache; trạng thái ok) và **bước 5 — validator, giám khảo**:

| Phim | Câu LLM viết | Nguồn | Validator | Độ giống | Giám khảo |
|---|---|---|---|---:|---|
| Cast Away (Two-Disc Special Edition) | You previously enjoyed the drama Forrest Gump directed by Robert Zemeckis, and Cast Away is also a drama directed by him. | E2 E4 H3 | giữ | 0.65 | supported |
| Cast Away (Two-Disc Special Edition) | A reviewer notes that Cast Away is linked to Tom Hanks's success in Forrest Gump. | E5 H3 | giữ | 0.56 | supported |
| Fargo: Special Edition | You rated O Brother, Where Art Thou? highly, and Fargo is directed by the Coen brothers. | E9 H1 | giữ | 0.69 | supported |
| National Lampoon's Christmas Vacation (Special Edition) … | You enjoyed A Christmas Story, and National Lampoon's Christmas Vacation is a comedy focused on the Christmas holiday. | E13 E14 H4 | giữ | 0.63 | supported |

**Lời giải thích hiển thị cho user** (chỉ các câu validator giữ lại):

- *Cast Away (Two-Disc Special Edition)*: You previously enjoyed the drama Forrest Gump directed by Robert Zemeckis, and Cast Away is also a drama directed by him [E2][E4][H3]. A reviewer notes that Cast Away is linked to Tom Hanks's success in Forrest Gump [E5][H3].
- *Fargo: Special Edition*: You rated O Brother, Where Art Thou? highly, and Fargo is directed by the Coen brothers [E9][H1].
- *National Lampoon's Christmas Vacation (Special Edition) [DVD]*: You enjoyed A Christmas Story, and National Lampoon's Christmas Vacation is a comedy focused on the Christmas holiday [E13][E14][H4].

**Đối chứng — giải thích tự do** (không bắt buộc trích dẫn, cùng thông tin):

- *Cast Away (Two-Disc Special Edition)*: You loved Forrest Gump, so you will likely appreciate another powerful drama directed by Robert Zemeckis. *(supported)* It shares the emotionally resonant storytelling found in your top-rated dramatic films. *(unsupported)*
- *Fargo: Special Edition*: Since you rated O Brother, Where Art Thou? 5/5, you will enjoy this acclaimed comedy directed by the Coen brothers. *(supported)* It features brilliant cinematography by Roger Deakins, who also worked on your favorite Coen brothers film. *(supported)*
- *National Lampoon's Christmas Vacation (Special Edition) [DVD]*: Given your love for classic holiday favorites like A Christmas Story, this festive comedy is a natural fit. *(supported)* It delivers the lighthearted seasonal humor that you frequently enjoy. *(unsupported)*

### Record 5 (ngẫu nhiên) — user `AFAG6VYWRCE2ATRT7YMZ3SR5TBWQ`

- **Query:** `as_of` = 1478733748000 (2016-11-09); target `B0152ZY7KQ` — *Directors: Greg Nicotero, Michael E. Satrazemis, David Boyd. Cast: Norman Reedus : …*, rating 5★.
- **Kết quả:** target **ở hạng 3** (tính theo cùng phim).

**Bước 1 — Hồ sơ.** 4 phim đã xem trước `as_of`; 4 phim thích (≥ 4★), 0 phim không thích (≤ 2★); 0 phim không có metadata. Bằng chứng lịch sử đưa cho LLM:

- [H1] 4★ Starring: Josh Duhamel Hilary Duff Jesse McCartney Directors: Olga Lopato Producers: Armen Manasaryan Georgi …
- [H2] 5★ Pixels Starring: Adam Sandler Kevin James Michelle Monaghan Directors: Chris Columbus Producers: Adam Sandler …
- [H3] 5★ Frozen Starring: Kristen Bell Idina Menzel Jonathan Groff Directors: Chris Buck Jennifer Lee Producers: …
- [H4] 5★ The Walking Dead Starring: Andrew Lincoln Norman Reedus Steven Yeun Directors: Greg Nicotero Michael …

**Bước 2 — Fusion top 10** (nhánh tìm ra phim, điểm fusion):

| Hạng | Phim | Nhánh | Điểm |
|---:|---|---|---:|
| 1 | Directors: Tony Tang. Cast: Rob Schneider, Josh Duhamel, Tom Skerritt | itemknn | 1.677 |
| 2 | Ant-Man (Plus Bonus Features) | itemknn | 1.519 |
| 3 | Directors: Greg Nicotero, Michael E. Satrazemis, David Boyd. Cast: Norman Reedus : … ✅ target | itemknn | 1.494 |
| 4 | The Stinky & Dirty Show | itemknn | 1.482 |
| 5 | MARVEL'S AVENGERS: AGE OF ULTRON | itemknn | 1.476 |
| 6 | Sea Level | itemknn | 1.439 |
| 7 | Jurassic World | itemknn | 1.428 |
| 8 | The Walking Dead | itemknn | 1.415 |
| 9 | Directors: Collin Tsandilis, Emmanuelle Gignac, Shi Chang. Cast: Lara Jill Miller, Jessica … | itemknn | 1.406 |
| 10 | Directors: Don Hall, Chris Williams. Cast: T.J. Miller, Jamie Chung, Maya Rudolph | itemknn | 1.365 |

**Bước 3 — Bằng chứng.** 36 mục cho top 10 (18 metadata, 18 review). Của top 3:

- *Directors: Tony Tang. Cast: Rob Schneider, Josh Duhamel, Tom Skerritt*: [E1] Directors: Tony Tang. Cast: Rob Schneider, Josh Duhamel, Tom Skerritt; [E2] When you have seen and loved Pixar's Cars or Planes, Wings pales …; [E3] One of the most enjoyable kid friendly show ever, music and graphics …
- *Ant-Man (Plus Bonus Features)*: [E4] title: Ant-Man (Plus Bonus Features); [E5] Directors: Peyton Reed. Cast: Paul Rudd, Michael Douglas, Evangeline Lilly; [E6] This movie is by far, the weakest of the recent entries in …; [E7] I'm a fan of most of the recent movies Marvel has brought …
- *Directors: Greg Nicotero, Michael E. Satrazemis, David Boyd. Cast: Norman Reedus : …*: [E8] Directors: Greg Nicotero, Michael E. Satrazemis, David Boyd. Cast: Norman Reedus : …; [E9] As the series has evolved, the overall quality of the show has …; [E10] I love The Walking Dead, however, Season 6 went south and never …

**Bước 4 — LLM giải thích có trích dẫn** (1113 token vào, 354 token ra, cache; trạng thái ok) và **bước 5 — validator, giám khảo**:

| Phim | Câu LLM viết | Nguồn | Validator | Độ giống | Giám khảo |
|---|---|---|---|---:|---|
| Directors: Tony Tang. Cast: Rob Schneider, … | Since you enjoyed watching Josh Duhamel, you may like this movie which also stars Josh Duhamel. | E1 H1 | giữ | 0.44 | supported |
| Directors: Tony Tang. Cast: Rob Schneider, … | A reviewer mentions that the music and graphics get an A+ and the characters and animation are very unique. | E3 | giữ | 0.77 | supported |
| Ant-Man (Plus Bonus Features) | A reviewer notes that the movie features a light-hearted tone with a sprinkling of comedy and entertaining action. | E6 E7 | giữ | 0.42 | partial |
| Directors: Greg Nicotero, Michael E. Satrazemis, … | Because you rated The Walking Dead 5/5, you might enjoy this continuation since it shares directors and cast members like Norman Reedus. | E8 H4 | giữ | 0.57 | supported |

**Lời giải thích hiển thị cho user** (chỉ các câu validator giữ lại):

- *Directors: Tony Tang. Cast: Rob Schneider, Josh Duhamel, Tom Skerritt*: Since you enjoyed watching Josh Duhamel, you may like this movie which also stars Josh Duhamel [E1][H1]. A reviewer mentions that the music and graphics get an A+ and the characters and animation are very unique [E3].
- *Ant-Man (Plus Bonus Features)*: A reviewer notes that the movie features a light-hearted tone with a sprinkling of comedy and entertaining action [E6][E7].
- *Directors: Greg Nicotero, Michael E. Satrazemis, David Boyd. Cast: Norman Reedus : …*: Because you rated The Walking Dead 5/5, you might enjoy this continuation since it shares directors and cast members like Norman Reedus [E8][H4].

**Đối chứng — giải thích tự do** (không bắt buộc trích dẫn, cùng thông tin):

- *Directors: Tony Tang. Cast: Rob Schneider, Josh Duhamel, Tom Skerritt*: You may enjoy this film because it features Josh Duhamel, whom you've watched before. *(supported)* It offers a family-friendly animated adventure similar to your highly rated picks. *(supported)*
- *Ant-Man (Plus Bonus Features)*: This movie delivers the action and light-hearted comedy found in other blockbusters you like. *(supported)* It features engaging superhero entertainment with a strong ensemble cast. *(unsupported)*
- *Directors: Greg Nicotero, Michael E. Satrazemis, David Boyd. Cast: Norman Reedus : …*: Since you love The Walking Dead, this item shares the same directors and cast members. *(supported)* It dives back into the intense drama and thrilling environment of the series. *(unsupported)*

### Record 6 (có target trong top 10) — user `AEGX5S2NAIXHZ7KVAIGZMM2YVBYA`

- **Query:** `as_of` = 1619135027497 (2021-04-22); target `B08XB281CB` — *Land*, rating 5★.
- **Kết quả:** target **ở hạng 2** (tính theo cùng phim).

**Bước 1 — Hồ sơ.** 7 phim đã xem trước `as_of`; 7 phim thích (≥ 4★), 0 phim không thích (≤ 2★); 0 phim không có metadata. Bằng chứng lịch sử đưa cho LLM:

- [H1] 5★ Wild Mountain Thyme [DVD] Starring: Emily Blunt Jamie Dornan Jon Hamm Directors: John Patrick …
- [H2] 5★ Alone With Chrissie Hynde Actors: Hynde, Chrissie Director: Nicola Roberts Studio: Eagle Rock Ent …
- [H3] 5★ Gremlins Starring: Zach Galligan Phoebe Cates Hoyt Axton Directors: Joe Dante Producers: Michael Finnell …
- [H4] 5★ Hitler's Eastern Front: Army Group South Directors: Paul Dunn Producers: Paul Dunn Audio languages: …
- [H5] 5★ Directors: Wain Fimeri Studio: Fisher Klingenstein Films Subtitles: None available
- [H6] 5★ Fort Utah Starring: John Ireland Virginia Mayo Directors: Lesley Selander Producers: A.C. Lyles Studio: …
- … và 1 mục khác

**Bước 2 — Fusion top 10** (nhánh tìm ra phim, điểm fusion):

| Hạng | Phim | Nhánh | Điểm |
|---:|---|---|---:|
| 1 | Beside Bowie: The Mick Ronson Story [Blu-ray + DVD] | itemknn | 1.331 |
| 2 | Land ✅ target | itemknn | 1.258 |
| 3 | Directors: Adriana Trigiani. Cast: Craig Ferguson, Kathie Lee Gifford, Ford Kiernan | itemknn | 1.160 |
| 4 | Ride the Pink Horse | itemknn | 1.047 |
| 5 | Little Feat: Skin It Back | itemknn | 1.044 |
| 6 | ROUND MIDNIGHT (DVD) | itemknn | 1.036 |
| 7 | Blade Runner: Black Lotus | itemknn | 1.035 |
| 8 | Live From The Artists Den | itemknn | 1.029 |
| 9 | Chicago: The Terry Kath Experience - Special Edition | itemknn | 1.027 |
| 10 | Michael Schenker Group - Rockpalast | itemknn | 1.026 |

**Bước 3 — Bằng chứng.** 50 mục cho top 10 (32 metadata, 18 review). Của top 3:

- *Beside Bowie: The Mick Ronson Story [Blu-ray + DVD]*: [E1] title: Beside Bowie: The Mick Ronson Story [Blu-ray + DVD]; [E2] genres: TV, Rock, Music Video & Concerts; [E3] description: Product Description From Emperor Media Ltd. And acclaimed producer/director Jon Brewer, …; [E4] Directors: Jon Brewer. Cast: David Bowie, Mick Ronson, Angie Bowie, Ian Hunter; [E5] I think this is a really good video, but it certainly could …; [E6] As David Bowie's lead guitarist, arranger and band leader from 1970-1973 Mick …
- *Land*: [E7] title: Land; [E8] Directors: Robin Wright. Cast: Sarah Dawn Pledge, Robin Wright, Demián Bichir; [E9] I am sad that I could watch this just 1 time for …; [E10] I was really disappointed with this movie. It was very very slow …
- *Directors: Adriana Trigiani. Cast: Craig Ferguson, Kathie Lee Gifford, Ford Kiernan*: [E11] Directors: Adriana Trigiani. Cast: Craig Ferguson, Kathie Lee Gifford, Ford Kiernan; [E12] A story of deferred dreams and ambitions and a second chance at …; [E13] I love Craig Ferguson and I like Elizabeth Hurley, so I thought …

**Bước 4 — LLM giải thích có trích dẫn** (1395 token vào, 174 token ra, cache; trạng thái ok) và **bước 5 — validator, giám khảo**:

| Phim | Câu LLM viết | Nguồn | Validator | Độ giống | Giám khảo |
|---|---|---|---|---:|---|
| Beside Bowie: The Mick Ronson Story … | This music documentary features rock history and artists like David Bowie, connecting to your interest in music features. | E1 E2 E3 E4 E6 H2 | giữ | 0.54 | supported |
| Land | (không có câu nào) |  | — | 0.00 | — |
| Directors: Adriana Trigiani. Cast: Craig Ferguson, … | (không có câu nào) |  | — | 0.00 | — |

**Lời giải thích hiển thị cho user** (chỉ các câu validator giữ lại):

- *Beside Bowie: The Mick Ronson Story [Blu-ray + DVD]*: This music documentary features rock history and artists like David Bowie, connecting to your interest in music features [E1][E2][E3][E4][E6][H2].
- *Land*: (không có giải thích — evidence_unavailable)
- *Directors: Adriana Trigiani. Cast: Craig Ferguson, Kathie Lee Gifford, Ford Kiernan*: (không có giải thích — evidence_unavailable)

**Đối chứng — giải thích tự do** (không bắt buộc trích dẫn, cùng thông tin):

- *Beside Bowie: The Mick Ronson Story [Blu-ray + DVD]*: You will likely enjoy this rock documentary given your appreciation for music performance and artist-focused specials. *(supported)* It explores the incredible life and musical contributions of iconic guitarist Mick Ronson with narration by David Bowie. *(supported)*
- *Land*: This dramatic feature offers striking scenic landscapes that capture the beauty of the outdoors. *(supported)* It is directed by and stars Robin Wright in a deeply introspective story. *(partial)*
- *Directors: Adriana Trigiani. Cast: Craig Ferguson, Kathie Lee Gifford, Ford Kiernan*: This film features well-known entertainment personalities like Craig Ferguson in a story about second chances. *(supported)* It focuses on themes of deferred dreams and finding unexpected romance. *(supported)*

### Record 7 (có target trong top 10) — user `AHBCPWX5TQVFOOQZN4FBRST6YSRQ`

- **Query:** `as_of` = 1525522721675 (2018-05-05); target `B076B7WYHT` — *Victoria & Abdul [Blu-ray]*, rating 3★.
- **Kết quả:** target **ở hạng 5** (tính theo cùng phim).

**Bước 1 — Hồ sơ.** 33 phim đã xem trước `as_of`; 9 phim thích (≥ 4★), 9 phim không thích (≤ 2★); 0 phim không có metadata. Bằng chứng lịch sử đưa cho LLM:

- [H1] 4★ The Zookeeper's Wife [Blu-ray] Starring: Jessica Chastain Johan Heldenbergh Michael McElhatton Directors: Niki Caro … — review của user: "Nice movie. Sound went loud, quiet, loud, quiet. Sound inconsistencies was a real bother."
- [H2] 5★ Z: The Beginning of Everything Starring: Christina Ricci David Hoflin Directors: Tim Blake Nelson … — review của user: "Good acting. Good scenery and costumes. Very well done...look forward to more."
- [H3] 5★ Goliath Starring: Billy Bob Thornton William Hurt Maria Bello Directors: Lawrence Trilling Alik Sakharov … — review của user: "Great acting. Great storyline a bit redundant of other lawyer vs big company books...but good. Tension."
- [H4] 4★ Mary and Martha Starring: Hilary Swank Brenda Blethyn Sam Claflin Directors: Phillip Noyce Producers: …
- [H5] 5★ Directors: Alan Poul Greg Mottola Anthony Hemingway Lesli Linka Glatter Jeremy Podeswa Alex Graves …
- [H6] 4★ Starring: Brendan Coyle Claudia Blakley Dawn French Directors: Susan Tully Patrick Lau Alan Grint … — review của user: "Well done. Well developed characters...with many focal points and story lines. No predictable outcome. I watch this when I want …"
- … và 12 mục khác

**Bước 2 — Fusion top 10** (nhánh tìm ra phim, điểm fusion):

| Hạng | Phim | Nhánh | Điểm |
|---:|---|---|---:|
| 1 | Directors: Sean Anders. Cast: Will Ferrell, Mark Wahlberg, Linda Cardellini | itemknn | 1.926 |
| 2 | Jumanji: Welcome to the Jungle [4K UHD + Blu-ray] | itemknn | 1.872 |
| 3 | Directors: Trish Sie. Cast: Anna Kendrick, Rebel Wilson, Brittany Snow | itemknn | 1.809 |
| 4 | Directors: Jonathan Levine. Cast: Amy Schumer, Goldie Hawn, Joan Cusack | itemknn | 1.799 |
| 5 | Victoria & Abdul [Blu-ray] ✅ target | itemknn | 1.795 |
| 6 | Directors: Theodore Melfi. Cast: Taraji P. Henson, Janelle Monae, Kevin Costner | itemknn | 1.780 |
| 7 | Directors: Marc Webb. Cast: Mckenna Grace, Chris Evans, Lindsay Duncan | itemknn | 1.739 |
| 8 | Directors: Kenneth Branagh. Cast: Kenneth Branagh, Penélope Cruz, Johnny Depp | itemknn | 1.714 |
| 9 | Directors: Scott Cooper. Cast: Christian Bale, Rosamund Pike, Wes Studi | itemknn | 1.706 |
| 10 | The Greatest Showman | itemknn | 1.706 |

**Bước 3 — Bằng chứng.** 33 mục cho top 10 (13 metadata, 20 review). Của top 3:

- *Directors: Sean Anders. Cast: Will Ferrell, Mark Wahlberg, Linda Cardellini*: [E1] Directors: Sean Anders. Cast: Will Ferrell, Mark Wahlberg, Linda Cardellini; [E2] I wouldn't watch it again. Some of Will Ferrell's movies are hit …; [E3] Surprised by laughter. Both husband and I enjoyed this. I don't usually …
- *Jumanji: Welcome to the Jungle [4K UHD + Blu-ray]*: [E4] title: Jumanji: Welcome to the Jungle [4K UHD + Blu-ray]; [E5] Directors: Jake Kasdan. Cast: Dwayne Johnson, Kevin Hart, Jack Black; [E6] My kids were so excited to see this movie (ages 7 & …; [E7] It was cute. It wasn't as entertaining as I thought it would …
- *Directors: Trish Sie. Cast: Anna Kendrick, Rebel Wilson, Brittany Snow*: [E8] Directors: Trish Sie. Cast: Anna Kendrick, Rebel Wilson, Brittany Snow; [E9] Pitch Perfect 3 was a lovely reminder of what's important in life: …; [E10] Worst movie of the year! Maybe worst of all time! There is …

**Bước 4 — LLM giải thích có trích dẫn** (1650 token vào, 253 token ra, cache; trạng thái ok) và **bước 5 — validator, giám khảo**:

| Phim | Câu LLM viết | Nguồn | Validator | Độ giống | Giám khảo |
|---|---|---|---|---:|---|
| Directors: Sean Anders. Cast: Will Ferrell, … | A reviewer says that some things were genuinely funny in the movie. | E2 | giữ | 0.44 | supported |
| Jumanji: Welcome to the Jungle [4K … | A reviewer mentions that the movie has many funny scenes and situations. | E6 | giữ | 0.50 | supported |
| Directors: Trish Sie. Cast: Anna Kendrick, … | A reviewer says the film is a lovely reminder of what is important in life, such as having fun and being loyal to friends. | E9 | giữ | 0.46 | supported |

**Lời giải thích hiển thị cho user** (chỉ các câu validator giữ lại):

- *Directors: Sean Anders. Cast: Will Ferrell, Mark Wahlberg, Linda Cardellini*: A reviewer says that some things were genuinely funny in the movie [E2].
- *Jumanji: Welcome to the Jungle [4K UHD + Blu-ray]*: A reviewer mentions that the movie has many funny scenes and situations [E6].
- *Directors: Trish Sie. Cast: Anna Kendrick, Rebel Wilson, Brittany Snow*: A reviewer says the film is a lovely reminder of what is important in life, such as having fun and being loyal to friends [E9].

**Đối chứng — giải thích tự do** (không bắt buộc trích dẫn, cùng thông tin):

- *Directors: Sean Anders. Cast: Will Ferrell, Mark Wahlberg, Linda Cardellini*: You tend to enjoy well-paced comedies when watching with family. *(unsupported)* Other viewers found this sequel surprisingly funny and engaging. *(supported)*
- *Jumanji: Welcome to the Jungle [4K UHD + Blu-ray]*: You enjoy movies with entertaining adventure elements and popular casts. *(unsupported)* This film features lighthearted humor that many family audiences appreciate. *(unsupported)*
- *Directors: Trish Sie. Cast: Anna Kendrick, Rebel Wilson, Brittany Snow*: You appreciate character-driven stories with themes of loyalty and friendship. *(partial)* Fans of the earlier films praised the uplifting and fun musical journey. *(unsupported)*

### Record 8 (có target trong top 10) — user `AGJPD3G6P2XGNRSYTVMUB3DEVK4A`

- **Query:** `as_of` = 1516052412499 (2018-01-15); target `B003ZKBELG` — *Twilight (Single-Disc Edition)*, rating 5★.
- **Kết quả:** target **ở hạng 1** (tính theo cùng phim).

**Bước 1 — Hồ sơ.** 5 phim đã xem trước `as_of`; 5 phim thích (≥ 4★), 0 phim không thích (≤ 2★); 0 phim không có metadata. Bằng chứng lịch sử đưa cho LLM:

- [H1] 5★ The Twilight Saga: Breaking Dawn - Part 2 Actors: Kristen Stewart, Robert Pattinson, Taylor …
- [H2] 5★ The Twilight Saga: Breaking Dawn - Part 1 (Two-Disc Special Edition) Genre: Drama Actors: …
- [H3] 5★ The Twilight Saga: New Moon (Single-Disc Edition) Genre: Action-Fantasy, Drama Actors: Kristen Stewart, Robert …
- [H4] 5★ The Twilight Saga: Eclipse (Single-Disc Edition) Genre: Drama Actors: Robert Pattinson, Kristen Stewart, Taylor …
- [H5] 5★ Country Classics: A Tapestry Of Our Musical Heritage Genre: Christian, Music Video & Concerts …

**Bước 2 — Fusion top 10** (nhánh tìm ra phim, điểm fusion):

| Hạng | Phim | Nhánh | Điểm |
|---:|---|---|---:|
| 1 | Twilight (Single-Disc Edition) ✅ target | bm25+itemknn | 2.002 |
| 2 | The Hobbit: An Unexpected Journey (Two-Disc Special Edition) | itemknn | 1.150 |
| 3 | The Hunger Games [DVD] | itemknn | 1.134 |
| 4 | The Hunger Games: Catching Fire [DVD] | itemknn | 1.127 |
| 5 | Harry Potter and the Deathly Hallows, Part 1 | itemknn | 1.126 |
| 6 | Harry Potter and the Deathly Hallows, Part 2 | itemknn | 1.125 |
| 7 | Dark Shadows | itemknn | 1.115 |
| 8 | Remember Me | itemknn | 1.113 |
| 9 | Underworld: Awakening | itemknn | 1.104 |
| 10 | Water for Elephants | bm25+itemknn | 1.100 |

**Bước 3 — Bằng chứng.** 59 mục cho top 10 (39 metadata, 20 review). Của top 3:

- *Twilight (Single-Disc Edition)*: [E1] title: Twilight (Single-Disc Edition); [E2] genres: Musicals & Performing Arts, Musicals, Action-Fantasy, Drama; [E3] description: Single Disc Version. Movie Only.; [E4] Directors: Catherine Hardwicke. Cast: Kristen Stewart, Robert Pattinson, Taylor Lautner, Billy Burke, …; [E5] Will somebody please tell me why I watch this movie every time …; [E6] I'm not exactly the target demographic for the TWILIGHT films - or …
- *The Hobbit: An Unexpected Journey (Two-Disc Special Edition)*: [E7] title: The Hobbit: An Unexpected Journey (Two-Disc Special Edition); [E8] genres: Science Fiction & Fantasy, Fantasy, Kids & Family, Action & Adventure; [E9] description: The adventure follows the journey of title character Bilbo Baggins, who …; [E10] Directors: Peter Jackson. Cast: Ian McKellen, Martin Freeman, Richard Armitage, James Nesbitt, …; [E11] First, I admit that I'm a fussy purist with these things. I …; [E12] As a child, I laid awake reading and re-reading Tolkien's 'The Hobbit' …
- *The Hunger Games [DVD]*: [E13] title: The Hunger Games [DVD]; [E14] genres: Action & Adventure; [E15] description: Amazon.com Building on her performance as a take-no-prisoners teenager in Winter's …; [E16] This is a guy's opinion, first of all. I never read the …; [E17] I am trying to keep my review as unbiased as I can …

**Bước 4 — LLM giải thích có trích dẫn** (1566 token vào, 293 token ra; trạng thái ok) và **bước 5 — validator, giám khảo**:

| Phim | Câu LLM viết | Nguồn | Validator | Độ giống | Giám khảo |
|---|---|---|---|---:|---|
| Twilight (Single-Disc Edition) | You will likely enjoy Twilight because it features the same main cast members as The Twilight Saga series you rated highly. | E4 H1 H2 H3 H4 | giữ | 0.60 | supported |
| Twilight (Single-Disc Edition) | A reviewer notes that the film is part of the Twilight franchise. | E6 H1 H2 H3 H4 | giữ | 0.61 | unsupported |
| The Hobbit: An Unexpected Journey (Two-Disc … | The Hobbit offers an epic fantasy adventure with dwarves, goblins, and wizards. | E9 H3 | giữ | 0.62 | supported |
| The Hobbit: An Unexpected Journey (Two-Disc … | A reviewer mentions reading Tolkien's original book many times as a child. | E12 | giữ | 0.66 | supported |
| The Hunger Games [DVD] | The Hunger Games is an action-oriented adaptation of a young adult bestseller set in a dystopian future. | E15 | giữ | 0.50 | supported |
| The Hunger Games [DVD] | A reviewer notes that the movie is based on a novel series. | E17 | giữ | 0.51 | supported |

**Lời giải thích hiển thị cho user** (chỉ các câu validator giữ lại):

- *Twilight (Single-Disc Edition)*: You will likely enjoy Twilight because it features the same main cast members as The Twilight Saga series you rated highly [E4][H1][H2][H3][H4]. A reviewer notes that the film is part of the Twilight franchise [E6][H1][H2][H3][H4].
- *The Hobbit: An Unexpected Journey (Two-Disc Special Edition)*: The Hobbit offers an epic fantasy adventure with dwarves, goblins, and wizards [E9][H3]. A reviewer mentions reading Tolkien's original book many times as a child [E12].
- *The Hunger Games [DVD]*: The Hunger Games is an action-oriented adaptation of a young adult bestseller set in a dystopian future [E15]. A reviewer notes that the movie is based on a novel series [E17].

**Đối chứng — giải thích tự do** (không bắt buộc trích dẫn, cùng thông tin):

- *Twilight (Single-Disc Edition)*: You loved all the other installments in The Twilight Saga, making this first film a perfect match for your collection. *(supported)* It features the same main cast and romantic fantasy themes that you thoroughly enjoyed in the later sequels. *(supported)*
- *The Hobbit: An Unexpected Journey (Two-Disc Special Edition)*: It shares the epic fantasy and action-adventure elements found in your highly rated Twilight Saga movies. *(supported)* You will enjoy following a grand quest featuring magical creatures and high-stakes journeys. *(supported)*
- *The Hunger Games [DVD]*: This film adapts a bestselling young adult novel into a dramatic, high-stakes cinematic experience. *(supported)* It appeals to fans of intense dystopian adventures and character-driven stories similar to your favorite fantasy series. *(supported)*

### Record 9 (có target trong top 10) — user `AHL5KJJA4B65AGV735VNIXIFUFWQ`

- **Query:** `as_of` = 1632538406640 (2021-09-25); target `B09D2N1H3C` — *Directors: Billy Bob Thornton, Lawrence Trilling, Derek Johansen. Cast: Billy Bob Thornton, …*, rating 5★.
- **Kết quả:** target **ở hạng 3** (tính theo cùng phim).

**Bước 1 — Hồ sơ.** 18 phim đã xem trước `as_of`; 13 phim thích (≥ 4★), 5 phim không thích (≤ 2★); 0 phim không có metadata. Bằng chứng lịch sử đưa cho LLM:

- [H1] 5★ Sneaky Pete Starring: Marin Ireland Shane Mcrae Giovanni Ribisi Directors: Rosemary Rodriguez Sarah Pia … — review của user: "I was a huge fan of the series and then all of a sudden it got cancelled. I hope they'll …"
- [H2] 5★ Jaws [Blu-ray] Genre: Jaws One, Drama, Horror, DVD Movie, Blu-ray Movie, Jaws I, Action … — review của user: "My family migrated to Martha's Vineyard in the late 1800's and continues to grow today. JAWS is a very special …"
- [H3] 5★ Starring: Paul Newman Tom Cruise Mary Mastrantonio Directors: Martin Scorses Producers: Irving Axelrad Barbara … — review của user: "I really liked how this movie was filmed and set up. They did a really great job portraying the pool …"
- [H4] 5★ Hunted Starring: Melissa George Adam Rayner Adewale Akinnuoye-Agbaje Directors: S.J. Clarkson Daniel Percival Alrick … — review của user: "I was enjoying watching the series and then it ended, I'm hoping it'll get picked up down the road in …"
- [H5] 5★ Starring: John Krasinski Wendell Pierce Noomi Rapace Directors: Jann Turner Patricia Riggen Daniel Sackheim … — review của user: "A series usually needs around 3 seasons to iron out the kinks and this series is a solid one so …"
- [H6] 5★ Content advisory: Nudity violence substance use alcohol use smoking foul language sexual content Audio … — review của user: "I went to college in Boston and lived in Somerville and I'm very familiar with Winter Hill and Broadway.."
- … và 12 mục khác

**Bước 2 — Fusion top 10** (nhánh tìm ra phim, điểm fusion):

| Hạng | Phim | Nhánh | Điểm |
|---:|---|---|---:|
| 1 | Sneaky Pete | bm25+itemknn | 1.352 |
| 2 | The Tomorrow War | itemknn+popularity | 1.277 |
| 3 | Directors: Billy Bob Thornton, Lawrence Trilling, Derek Johansen. Cast: Billy Bob Thornton, … ✅ target | itemknn | 1.257 |
| 4 | Directors: Ernest Dickerson, Alex Zakrzewski, Zetna Fuentes, Daisy von Scherler Mayer, Patrick … | itemknn | 1.252 |
| 5 | Directors: Thomas Vincent, Sam Hill, Stephen Surjik, Christine Moore, Norberto Barba, Omar … | itemknn | 1.246 |
| 6 | Honest Thief [DVD] | itemknn | 1.233 |
| 7 | Directors: Scott Ellis. Cast: Alex Borstein, Marin Hinkle, Rachel Brosnahan | itemknn | 1.167 |
| 8 | Directors: John Butler, Alicia MacDonald, Stephen Merchant. Cast: Charles Babalola, Rhianne Barreto, … | itemknn | 1.163 |
| 9 | MORTAL BD + DGTL [Blu-ray] | itemknn | 1.161 |
| 10 | Green Zone | itemknn | 1.160 |

**Bước 3 — Bằng chứng.** 35 mục cho top 10 (19 metadata, 16 review). Của top 3:

- *Sneaky Pete*: [E1] title: Sneaky Pete; [E2] Directors: Jon Avnet, Laura Innes, Michael Dinner, Adam Arkin, Clare Kilner, Seth …; [E3] Don't listen to the negative reviews. Season three is great. My only …; [E4] I fell in love with this series... it is smart, suspenseful, compassionate, …
- *The Tomorrow War*: [E5] title: The Tomorrow War; [E6] genres: Suspense, Drama, Compelling, Frightening; [E7] description: Time travelers arrive from 2051 to deliver an urgent message: 30 …; [E8] Directors: Chris McKay. Cast: Chris Pratt, J.K. Simmons, Yvonne Strahovski; [E9] If Tom Cruise’s Edge of Tomorrow decided to have a Cochella lazy …; [E10] The Tomorrow War is astonishingly bad. So sorry to see Chris Pratt …
- *Directors: Billy Bob Thornton, Lawrence Trilling, Derek Johansen. Cast: Billy Bob Thornton, …*: [E11] Directors: Billy Bob Thornton, Lawrence Trilling, Derek Johansen. Cast: Billy Bob Thornton, …; [E12] The show writers simply ran out of ideas and added momentous segments …; [E13] I'm three and a half episodes in, and loving it. The tone …

**Bước 4 — LLM giải thích có trích dẫn** (2047 token vào, 231 token ra, cache; trạng thái ok) và **bước 5 — validator, giám khảo**:

| Phim | Câu LLM viết | Nguồn | Validator | Độ giống | Giám khảo |
|---|---|---|---|---:|---|
| Sneaky Pete | You previously enjoyed Sneaky Pete, and a reviewer notes that it features an incredible cast of character actors and actresses. | E3 H1 | giữ | 0.75 | supported |
| The Tomorrow War | (không có câu nào) |  | — | 0.00 | — |
| Directors: Billy Bob Thornton, Lawrence Trilling, … | You previously watched and rated Goliath starring Billy Bob Thornton, and this item features the same director and lead actor. | E11 H7 H8 | giữ | 0.60 | supported |

**Lời giải thích hiển thị cho user** (chỉ các câu validator giữ lại):

- *Sneaky Pete*: You previously enjoyed Sneaky Pete, and a reviewer notes that it features an incredible cast of character actors and actresses [E3][H1].
- *The Tomorrow War*: (không có giải thích — evidence_unavailable)
- *Directors: Billy Bob Thornton, Lawrence Trilling, Derek Johansen. Cast: Billy Bob Thornton, …*: You previously watched and rated Goliath starring Billy Bob Thornton, and this item features the same director and lead actor [E11][H7][H8].

**Đối chứng — giải thích tự do** (không bắt buộc trích dẫn, cùng thông tin):

- *Sneaky Pete*: You previously enjoyed watching Sneaky Pete and appreciated its cast and plot. *(partial)* This item matches your interest in smart, suspenseful drama series. *(partial)*
- *The Tomorrow War*: This suspenseful and compelling movie features action and drama elements. *(supported)* It aligns with your interest in thrilling cinematic adventures. *(supported)*
- *Directors: Billy Bob Thornton, Lawrence Trilling, Derek Johansen. Cast: Billy Bob Thornton, …*: You have previously given high ratings to Billy Bob Thornton's series. *(supported)* This recommendation shares the engaging noir tone and artistic nuances you enjoy. *(partial)*

### Record 10 (có target trong top 10) — user `AH7SKH7TUUWPVV4KXPQSLUMQNAKA`

- **Query:** `as_of` = 1653376858752 (2022-05-24); target `B09ML1QV1S` — *Directors: Thomas Vincent, Sam Hill, Stephen Surjik, Christine Moore, Norberto Barba, Omar …*, rating 4★.
- **Kết quả:** target **ở hạng 5** (tính theo cùng phim).

**Bước 1 — Hồ sơ.** 8 phim đã xem trước `as_of`; 7 phim thích (≥ 4★), 1 phim không thích (≤ 2★); 0 phim không có metadata. Bằng chứng lịch sử đưa cho LLM:

- [H1] 4★ The Tomorrow War Starring: Chris Pratt J.K. Simmons Yvonne Strahovski Directors: Chris McKay Producers: … — review của user: "i just still dont get why the first jump went wrong"
- [H2] 5★ Starring: Kingsley Ben-Adir Eli Goree Aldis Hodge Leslie Odom Jr Directors: Regina King Producers: … — review của user: "ITS HARD NOT TO BELIVE that alot of this is fictional"
- [H3] 5★ Contagion (Movie-Only Edition + UltraViolet Digital Copy) [Blu-ray] Starring: Matt Damon Kate Winslet Jude … — review của user: "i saw this 3 months B4 march 2020 covid was in china and i knew it would be big so …"
- [H4] 4★ Godzilla [Blu-ray] Genre: Action, Horror Actors: Matthew Broderick, Jean Reno, Maria Pitillo, Hank Azaria, …
- [H5] 5★ Sneaky Pete Starring: Marin Ireland Shane Mcrae Giovanni Ribisi Directors: Rosemary Rodriguez Sarah Pia … — review của user: "i think anyone who watches 15 min of one episode should not judge or write a review i watched all …"
- [H6] 5★ Starring: Kevin Conroy Efrem Zimbalist Jr. Loren Lester Directors: Kevin Altieri Boyd Kirkland Frank … — review của user: "i got this on amazon prime yes i love batman and season three is where they bring the true original …"
- … và 2 mục khác

**Bước 2 — Fusion top 10** (nhánh tìm ra phim, điểm fusion):

| Hạng | Phim | Nhánh | Điểm |
|---:|---|---|---:|
| 1 | Directors: Stefano Sollima. Cast: Michael B. Jordan, Jamie Bell, Jodie Turner-Smith | itemknn | 1.994 |
| 2 | Directors: Craig Brewer. Cast: Eddie Murphy, Arsenio Hall, Jermaine Fowler | itemknn | 1.859 |
| 3 | The Wheel of Time | itemknn | 1.839 |
| 4 | Directors: Antoine Fuqua, Ellen Kuras, MJ Bassett, Fred Toye, Tucker Gates, Sylvain … | itemknn | 1.768 |
| 5 | Directors: Thomas Vincent, Sam Hill, Stephen Surjik, Christine Moore, Norberto Barba, Omar … ✅ target | itemknn | 1.704 |
| 6 | Cinderella | itemknn | 1.574 |
| 7 | Tales of the Teenage Mutant Ninja Turtles Super Shredder | itemknn | 1.569 |
| 8 | Directors: Breck Eisner, Thomas Jane, Nick Gomez, Jeff Woolnough, Marisol Adler. Cast: … | itemknn | 1.560 |
| 9 | Directors: Wayne Yip, Charlotte Brändström, J.A. Bayona, Sanaa Hamri, Louise Hooper | itemknn | 1.545 |
| 10 | Outer Range | itemknn | 1.539 |

**Bước 3 — Bằng chứng.** 32 mục cho top 10 (16 metadata, 16 review). Của top 3:

- *Directors: Stefano Sollima. Cast: Michael B. Jordan, Jamie Bell, Jodie Turner-Smith*: [E1] Directors: Stefano Sollima. Cast: Michael B. Jordan, Jamie Bell, Jodie Turner-Smith; [E2] Being a Tom Clancy fan, I was ready for this to be …; [E3] I'm not sure who wrote the screenplay, directed this, or if they …
- *Directors: Craig Brewer. Cast: Eddie Murphy, Arsenio Hall, Jermaine Fowler*: [E4] Directors: Craig Brewer. Cast: Eddie Murphy, Arsenio Hall, Jermaine Fowler; [E5] I was expecting more Black folks GOOD, white folks BAAAD, but there …; [E6] Within the first 3 minutes you will be speechless from the disappointment. …
- *The Wheel of Time*: [E7] title: The Wheel of Time; [E8] Directors: Uta Briesewitz, Wayne Yip, Salli Richardson Whitfield, Ciaran Donnelly. Cast: Rosamund …; [E9] Who ever created the cast for this show knew nothing about the …; [E10] I was lucky enough to watch the first two episodes at an …

**Bước 4 — LLM giải thích có trích dẫn** (1561 token vào, 229 token ra, cache; trạng thái ok) và **bước 5 — validator, giám khảo**:

| Phim | Câu LLM viết | Nguồn | Validator | Độ giống | Giám khảo |
|---|---|---|---|---:|---|
| Directors: Stefano Sollima. Cast: Michael B. … | This movie features action elements directed by Stefano Sollima and starring Michael B. Jordan. | E1 | giữ | 0.67 | supported |
| Directors: Craig Brewer. Cast: Eddie Murphy, … | This comedy features a cast including Eddie Murphy and Arsenio Hall. | E4 | giữ | 0.73 | partial |
| The Wheel of Time | Based on a book series, a reviewer notes that the fast pace helps cover a lot of the world. | E10 | giữ | 0.31 | partial |

**Lời giải thích hiển thị cho user** (chỉ các câu validator giữ lại):

- *Directors: Stefano Sollima. Cast: Michael B. Jordan, Jamie Bell, Jodie Turner-Smith*: This movie features action elements directed by Stefano Sollima and starring Michael B. Jordan [E1].
- *Directors: Craig Brewer. Cast: Eddie Murphy, Arsenio Hall, Jermaine Fowler*: This comedy features a cast including Eddie Murphy and Arsenio Hall [E4].
- *The Wheel of Time*: Based on a book series, a reviewer notes that the fast pace helps cover a lot of the world [E10].

**Đối chứng — giải thích tự do** (không bắt buộc trích dẫn, cùng thông tin):

- *Directors: Stefano Sollima. Cast: Michael B. Jordan, Jamie Bell, Jodie Turner-Smith*: You might enjoy this action-heavy thriller given your positive rating for military sci-fi like The Tomorrow War. *(supported)* It features intense special ops infiltrations similar to high-stakes action movies you have watched. *(supported)*
- *Directors: Craig Brewer. Cast: Eddie Murphy, Arsenio Hall, Jermaine Fowler*: This comedy features major star power and high-profile production names you often look for in films. *(unsupported)* It is a widely discussed sequel release that attracts strong viewer opinions just like your favorite blockbusters. *(unsupported)*
- *The Wheel of Time*: As a fan of serialized dramas and multi-episode series like Sneaky Pete, you might appreciate this epic fantasy adaptation. *(supported)* Reviewers note that watching the initial episodes helps build a solid appreciation for this complex world. *(supported)*

## 3. Số liệu tổng hợp

| Nhóm | Số user | NDCG@10 | HR@10 |
|---|---:|---:|---:|
| random | 5 | 10,0% | 20,0% |
| with_target | 5 | 58,1% | 100,0% |

Giải thích (30 phim cần giải thích):

| Cấu hình | Số câu | Phim có giải thích | supported | partial | unsupported |
|---|---:|---:|---:|---:|---:|
| Tự do | 60 | 30 | 43 | 7 | 10 |
| Bắt buộc trích dẫn, trước validator | 34 | 27 | 30 | 3 | 1 |
| Bắt buộc trích dẫn, sau validator | 33 | 27 | 29 | 3 | 1 |

- Validator loại 1 câu: nhãn giám khảo {'supported': 1}; lý do: chỉ độ giống thấp.
- Theo ngưỡng độ giống: 0.0: 34 câu, supported 88,2%; 0.2: 34 câu, supported 88,2%; 0.3: 33 câu, supported 87,9%; 0.4: 30 câu, supported 90,0%; 0.5: 24 câu, supported 91,7%.
- Lỗi sinh: {'free': 0, 'cited': 0}.
- Token: giải thích có trích dẫn 16157 vào / 2872 ra; tự do 14316 / 2480; giám khảo 32123 / 11812.
- Độ trễ một lời gọi giải thích có trích dẫn (không tính cache): p50 1,6 s, p95 1,6 s.

## 4. Đánh giá tổng quan

*Viết tay sau khi đọc kết quả (18/10/2026). Chạy lại lệnh sẽ ghi đè cả file, kể cả mục này.*

**Kết luận chung:** luồng hồ sơ → fusion → bằng chứng → giải thích có trích dẫn → validator chạy trọn cho cả 10 record, không có lời gọi LLM nào lỗi; code sẵn sàng cho thí nghiệm lớn (200 user). Mẫu 10 user chỉ dùng để kiểm tra code và nhìn chất lượng định tính, **không** dùng để kết luận số liệu.

**Xếp hạng (fusion, không xếp lại).**

- 5 user ngẫu nhiên: 1/5 trúng (record 5, The Walking Dead ở hạng 3, chỉ trúng khi tính cùng phim vì target là một bản Prime khác). Tỉ lệ có target trong top 10 của lượt quét 5.000 user là 10,2%, khớp HR@10 cùng phim ~10,4% đo trên mẫu 50k test.
- Top 10 gần như hoàn toàn do ItemKNN tìm ra; BM25 chỉ góp mặt khi lịch sử có chuỗi phim rõ (Twilight, Water for Elephants) hoặc diễn viên chung. Gợi ý của các user trượt vẫn đúng gu (record 1: phim Giáng sinh, cổ điển nhẹ nhàng; record 2: anime/fantasy; record 4: Coen, Zemeckis).
- **Lỗi tìm ra nhờ smoke test và đã sửa:** ở lần chạy đầu, record 8 (fan Twilight) được gợi ý lại *Eclipse* và *Breaking Dawn Part 1* — hai phim user đã chấm 5★ — dưới phiên bản khác (bản 2 đĩa, bản DVD); chính LLM còn viết "đây là bản 2 đĩa của cùng bộ phim". Nguyên nhân: số lượt rating của các phiên bản lệch 2,8–3,3% (> ngưỡng 2%) và title viết khác ("Twilight Saga, The: …", "Part One"). Sửa `backend/data/editions.py`: cùng title chuẩn hóa thì cho lệch tới 5%, chỉ cùng người thì giữ 2%; chuẩn hóa "X, The:", "Part One" → "part 1", bỏ năm trong ngoặc. Báo cáo này là lần chạy sau khi sửa: top 10 của record 8 không còn phim đã xem (Hobbit, Hunger Games, Harry Potter…). Tác dụng phụ đo được: tỉ lệ target là phiên bản khác của phim đã xem tăng 0,08% → 0,12% (không đáng kể).

**Bằng chứng.** Mỗi user có 32–59 mục bằng chứng cho top 10 (metadata + tối đa 2 review/phim, đều viết trước `as_of`, không có review của chính user). Điểm yếu còn lại: nhiều phim Prime Video không có title (record 5, 7, 9, 10) nên chỉ hiển thị bằng tên đạo diễn/diễn viên; LLM vẫn giải thích được nhờ tên người nhưng người dùng khó nhận ra phim.

**Giải thích có trích dẫn so với tự do** (30 phim, giám khảo `gemini-3.5-flash`):

| | Tự do | Bắt buộc trích dẫn (sau validator) |
|---|---:|---:|
| Số câu | 60 | 33 |
| supported | 43 (72%) | 29 (88%) |
| unsupported | 10 (17%) | 1 (3%) |
| Phim có giải thích | 30/30 | 27/30 |

- Bắt buộc trích dẫn làm câu sai giảm rõ (17% → 3%), đổi lại **ít nói hơn** (33 so với 60 câu) và **từ chối giải thích 3/30 phim** khi không tìm được lý do có nguồn (Land, phim của Adriana Trigiani, The Tomorrow War). Từ chối là hành vi đúng thiết kế (`evidence_unavailable`), nhưng phải báo cùng tỉ lệ hỗ trợ để không "ăn gian" bằng cách im lặng.
- Câu sai của bản tự do chủ yếu là **gán sở thích không có trong lịch sử** ("You tend to enjoy well-paced comedies when watching with family", "the lighthearted seasonal humor that you frequently enjoy") — đúng loại lỗi đề tài muốn chặn.
- Mỗi câu có trích dẫn dẫn trung bình 2,1 nguồn; có câu dẫn cả H1–H4 dù chỉ cần một ("over-citation").

**Validator.**

- Kiểm tra ID, phim, tên người: không bắt lỗi nào ở lần này (JSON schema + prompt khiến LLM ít sai ID); các lỗi kiểu này đã gặp ở smoke test validation trước (dẫn E# của phim khác, chèn mã vào câu) và được xử lý.
- Lọc theo độ giống (ngưỡng 0,3) **chưa hữu ích**: loại 1 câu mà giám khảo chấm *supported* (độ giống 0,26) và giữ 1 câu *unsupported* ("A reviewer notes that the film is part of the Twilight franchise", độ giống 0,61). Độ giống câu–nguồn không phân biệt được "nói cùng chủ đề" với "nguồn thật sự nói vậy". Cần hiệu chỉnh ngưỡng trên mẫu validation lớn (bảng ngưỡng 0–0,5 ở mục 3) hoặc bỏ bước này.

**Điểm chất lượng cần cải thiện (định tính).**

1. Câu đúng nguồn nhưng **không phải lý do**: "A reviewer mentions reading Tolkien's original book many times as a child" (The Hobbit).
2. **Chọn ý khen trong review chê**: record 7, "a reviewer says that some things were genuinely funny" lấy từ review mở đầu "I wouldn't watch it again… the movie as a whole is sub-par". Giám khảo chấm *supported* vì câu đúng chữ; cần rubric "trung thực với ý chính của review" nếu muốn bắt lỗi này.
3. Giám khảo là LLM nên có thể chấm dễ: cần đối chiếu tay 30–50 câu (`runs/llm/smoke-10/manual_check.csv` đã xuất sẵn mẫu).

**Chi phí thực tế (10 user).** Giải thích có trích dẫn: ~1.600 token vào + ~290 token ra mỗi user, p50 1,6 s mỗi lời gọi; hệ thống cuối chỉ cần **1 lời gọi LLM/user**. Bản tự do và giám khảo chỉ dùng cho thí nghiệm (giám khảo ~3.200 vào + ~1.200 ra mỗi user, vì có suy nghĩ).

**Đề xuất trước khi chạy 200 user:** (1) chạy 50 user validation để chốt ngưỡng độ giống (hoặc bỏ); (2) thêm vào prompt: chỉ đưa lý do liên quan tới sở thích của user, không trích ý trái với ý chính của review, mỗi câu dẫn tối đa 2–3 nguồn; (3) giữ nguyên giám khảo và rubric để so sánh tự do vs. trích dẫn công bằng.

