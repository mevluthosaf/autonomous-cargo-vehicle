import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

# 1. OSM'de aranacak bölgenin adı (OSM veritabanındaki adıyla eşleşmeli)
# Eğer bu isimle bulamazsa, kampüsün enlem ve boylam sınırlarını (bbox) da verebiliriz.
place_name = "Selçuk Üniversitesi, Selçuklu, Konya, Turkey"

# 2. Sadece bisiklet yollarını çekecek filtre
# Eğer OSM üzerinde "cycleway"="lane" veya "cycleway"="track" gibi farklı
# etiketler kullandıysanız, bu filtreyi ona göre güncelleyebilirsiniz.
custom_filter = '["highway"="cycleway"]'

print(f"{place_name} bölgesi için bisiklet yolları indiriliyor, lütfen bekleyin...")

try:
    # 3. Belirtilen bölgedeki yolları indir ve Graf (Ağ) yapısına dönüştür
    # retain_all=True yapıyoruz ki birbiriyle kopuk çizilmiş yollar varsa silinmesin.
    #G = ox.graph_from_place(place_name, network_type='all', custom_filter=custom_filter, retain_all=True)
    # API'den değil, indirdiğimiz güncel dosyadan grafı oluştur
    G = ox.graph_from_xml("map.osm")

    print("Bisiklet yolu haritası başarıyla indirildi!")

    # Graf yapısal bilgilerini yazdır
    print(f"Toplam Kavşak/Dönüş Noktası (Node) Sayısı: {len(G.nodes)}")
    print(f"Toplam Yol Parçası (Edge) Sayısı: {len(G.edges)}")

    # 4. İndirilen Ağı Görselleştir
    # Düğümleri (Kavşakları) kırmızı, yolları mavi renkte çizdiriyoruz
    fig, ax = ox.plot_graph(G, node_color='r', node_size=15, edge_color='b', edge_linewidth=2, bgcolor='white')

    # İstersen bu grafı ileride internetsiz çalışmak üzere kaydedebilirsin:
    # ox.save_graphml(G, filepath="selcuk_bisiklet_yollari.graphml")
    # print("Harita çevrimdışı kullanım için kaydedildi.")

except Exception as e:
    print("Harita indirilirken bir hata oluştu.")
    print(
        "Sebepler: OSM henüz çizimlerinizi onaylayıp ana sunucuya yansıtmamış olabilir (bazen 1-2 saat sürer) veya bölge adı yanlış olabilir.")
    print(f"Hata Detayı: {e}")