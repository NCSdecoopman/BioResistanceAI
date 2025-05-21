# pipelines/run_test.py
import subprocess

def main():
    print("Lancement des tests unitaires...")
    result = subprocess.run(["pytest"], capture_output=True, text=True)

    print(result.stdout)
    if result.returncode != 0:
        print("Échec de certains tests.")
    else:
        print("Tous les tests ont réussi.")

if __name__ == "__main__":
    main()
