#include "functions.hpp"
#include "search.hpp"
#include <iomanip>
#include <iostream>
#include <vector>

long filter(Instance inst) {
  long legendaries = 0;
  inst.nextPack(1);
  for (int p = 1; p <= 3; p++) {
    Pack pack = packInfo(inst.nextPack(1));
    if (pack.type == Item::Arcana_Pack) {
      auto packContents = inst.nextArcanaPack(pack.size, 1);
      for (int x = 0; x < pack.size; x++) {
        if (packContents[x] == Item::The_Soul)
          legendaries++;
      }
    }
    if (pack.type == Item::Spectral_Pack) {
      auto packContents = inst.nextSpectralPack(pack.size, 1);
      for (int x = 0; x < pack.size; x++) {
        if (packContents[x] == Item::The_Soul)
          legendaries++;
      }
    }
  }
  return legendaries;
};

long filter_perkeo_observatory(Instance inst) {
  if (inst.nextVoucher(1) == Item::Telescope) {
    inst.activateVoucher(Item::Telescope);
    if (inst.nextVoucher(2) != Item::Observatory)
      return 0;
  } else
    return 0;
  int antes[5] = {1, 1, 2, 2, 2};
  for (int i = 0; i < 5; i++) {
    Pack pack = packInfo(inst.nextPack(antes[i]));
    std::vector<Item> packContents;
    if (pack.type == Item::Arcana_Pack) {
      packContents = inst.nextArcanaPack(pack.size, antes[i]);
    } else if (pack.type == Item::Spectral_Pack) {
      packContents = inst.nextSpectralPack(pack.size, antes[i]);
    } else
      continue;
    for (int x = 0; x < pack.size; x++) {
      if (packContents[x] == Item::The_Soul &&
          inst.nextJoker(ItemSource::Soul, antes[i], true).joker ==
              Item::Perkeo)
        return 1;
    }
  }
  return 0;
}

long filter_negative_tag(Instance inst) {
  // Note: If the score cutoff was passed as a variable, this code could be
  // significantly optimized
  int maxAnte = 20;
  int score = 0;
  for (int i = 2; i <= maxAnte; i++) {
    if (inst.nextTag(i) == Item::Negative_Tag)
      score++;
  }
  return score;
}

long filter_lucky(Instance inst) {
  for (int i = 0; i < 7; i++) {
    if (inst.random(RandomType::Lucky_Money) >= 1.0/15) {
      return 0;
    }
  }
  return 1;
}

long filter_suas_speedrun(Instance inst) {
  // First four cards in shop must include Mr. Bones, Merry Andy, and Luchador
  bool bones = false, andy = false, luchador = false;
  for (int i = 0; i < 4; i++) {
    ShopItem item = inst.nextShopItem(2);
    if (item.item == Item::Mr_Bones)
      bones = true;
    if (item.item == Item::Merry_Andy)
      andy = true;
    if (item.item == Item::Luchador)
      luchador = true;
  }
  if (!bones || !andy || !luchador)
    return 0;
  // Ante 1 must have a Coupon Tag
  inst.initLocks(1, false, true);
  bool coupon = false;
  for (int i = 0; i < 2; i++) {
    if (inst.nextTag(1) == Item::Coupon_Tag)
      coupon = true;
  }
  if (!coupon)
    return 1;
  // Ante 2 Boss must be The Wall
  inst.nextBoss(1);
  inst.initUnlocks(2, false);
  if (inst.nextBoss(2) != Item::The_Wall)
    return 2;
  return 3;
}

long filter_cavendish(Instance inst) {
  inst.initLocks(1, false, false);
  // Check for a Charm Tag (Arcana Pack)
  if (inst.nextTag(1) != Item::Charm_Tag)
    return 0;
  // Check for a Judgement within that pack
  std::vector<Item> packContents = inst.nextArcanaPack(5, 1);
  bool hasJudgement = false;
  for (int i = 0; i < 5; i++) {
    if (packContents[i] == Item::Judgement)
      hasJudgement = true;
  }
  if (!hasJudgement)
    return 1;
  // Check for Gros Michel
  if (inst.nextJoker(ItemSource::Judgement, 1, false).joker != Item::Gros_Michel)
    return 2;
  // Check for Gros Michel break
  if (inst.random(RandomType::Gros_Michel) >= 1.0/6)
    return 3;
  // Check for Cavendish in first shop
  if (inst.nextShopItem(1).item != Item::Cavendish || inst.nextShopItem(1).item != Item::Cavendish)
    return 4;
  // Check for Cavendish break
  if (inst.random(RandomType::Cavendish) < 1.0/1000)
    return 9999;
  return 5;
}

long filter_blank(Instance inst) { return 0; }

// These won't be permanent filters, just ones I sub in and out while JSON
// filters aren't ready yet
long filter_test(Instance inst) {
  // Four Fingers, Shortcut, and Smeared Joker in first two antes
  // (https://discord.com/channels/1325151824638120007/1326284714125955183)
  bool fingers = false;
  bool shortcut = false;
  bool smeared = false;
  // 4 chances in Ante 1, 6 chances in Ante 2, so no rerolling
  for (int i = 0; i < 4; i++) {
    ShopItem item = inst.nextShopItem(1);
    if (item.item == Item::Four_Fingers) {
      fingers = true;
    };
    if (item.item == Item::Shortcut) {
      shortcut = true;
    };
    if (item.item == Item::Smeared_Joker) {
      smeared = true;
    };
  }
  for (int i = 0; i < 6; i++) {
    ShopItem item = inst.nextShopItem(2);
    if (item.item == Item::Four_Fingers) {
      fingers = true;
    };
    if (item.item == Item::Shortcut) {
      shortcut = true;
    };
    if (item.item == Item::Smeared_Joker) {
      smeared = true;
    };
  }
  if (fingers && shortcut && smeared) {
    return 1;
  }
  return 0;
}

// Benchmark function
// Runs 1 billion seeds of perkeo observatory
// And prints total time and seeds per second
void benchmark() {
  long total = 0;
  long start = std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count();
  Search search(filter_perkeo_observatory, "IMMOLATE", 12, 1000000000);
  search.highScore = 10; // No output
  search.printDelay = 100000000000;
  search.search();
  long end = std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
  std::cout << "------LONGER TESTING------\n";
  std::cout << "Total time: " << end - start << "ms\n";
  std::cout << "Seeds per second: " << std::fixed << std::setprecision(0)
            << 1000000000 / ((end - start) / 1000.0) << "\n";
}

void benchmark_quick() {
  long total = 0;
  long start = std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count();
  Search search(filter_perkeo_observatory, "IMMOLATE", 12, 100000000);
  search.highScore = 10; // No output
  search.printDelay = 100000000000;
  search.search();
  long end = std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
  std::cout << "----PERKEO OBSERVATORY----\n";
  std::cout << "Total time: " << end - start << "ms\n";
  std::cout << "Seeds per second: " << std::fixed << std::setprecision(0)
            << 100000000 / ((end - start) / 1000.0) << "\n";
}

void benchmark_quick_lucky() {
  long total = 0;
  long start = std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count();
  Search search(filter_lucky, "IMMOLATE", 12, 100000000);
  search.highScore = 10; // No output
  search.printDelay = 100000000000;
  search.search();
  long end = std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
  std::cout << "-------LUCKY CARDS-------\n";
  std::cout << "Total time: " << end - start << "ms\n";
  std::cout << "Seeds per second: " << std::fixed << std::setprecision(0)
            << 100000000 / ((end - start) / 1000.0) << "\n";
}

void benchmark_single() {
  long total = 0;
  long start = std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count();
  Search search(filter_perkeo_observatory, "IMMOLATE", 1, 10000000);
  search.highScore = 10; // No output
  search.printDelay = 100000000000;
  search.search();
  long end = std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
  std::cout << "----SINGLE THREADED PO----\n";
  std::cout << "Total time: " << end - start << "ms\n";
  std::cout << "Seeds per second: " << std::fixed << std::setprecision(0)
            << 10000000 / ((end - start) / 1000.0) << "\n";
}

void benchmark_blank() {
  long total = 0;
  long start = std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count();
  Search search(filter_blank, "IMMOLATE", 12, 100000000);
  search.printDelay = 100000000000; // No output
  search.search();
  long end = std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
  std::cout << "-------BLANK FILTER-------\n";
  std::cout << "Total time: " << end - start << "ms\n";
  std::cout << "Seeds per second: " << std::fixed << std::setprecision(0)
            << 100000000 / ((end - start) / 1000.0) << "\n";
}

int main() {
  /*benchmark_single();
  benchmark_quick();
  benchmark_quick_lucky();
  benchmark_blank();
  benchmark();*/
  Search search(filter_cavendish, "11111J31", 8, 2318107019761);
  search.highScore = 5;
  search.printDelay = 2318107019761;
  search.search();
  return 1;
}